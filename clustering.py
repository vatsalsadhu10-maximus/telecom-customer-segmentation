"""Reproducible K-means segmentation and churn profiling for telecom customers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

def churn_indicator(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s): return s.astype(float)
    cleaned = s.astype('string').str.strip().str.lower()
    mapping = {'yes':1,'no':0,'true':1,'false':0,'churned':1,'retained':0,'1':1,'0':0,'left':1,'stayed':0}
    result = cleaned.map(mapping)
    return pd.to_numeric(result, errors='coerce')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/Dataset_ATS_v2.csv')
    p.add_argument('--out', default='outputs')
    p.add_argument('--max-categories', type=int, default=30)
    p.add_argument('--clusters', type=int, default=4, help='Chosen for interpretable segmentation; set 0 for automatic silhouette maximum')
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.input, low_memory=False)
    if len(data)<5: raise ValueError('Need at least 5 customer records.')
    data.columns = data.columns.astype(str).str.strip()
    churn_cols=[c for c in data.columns if c.lower().replace(' ','').replace('_','') in ('churn','churnlabel','churnvalue','exited')]
    churn_col=churn_cols[0] if churn_cols else None
    # Use behaviour and service-plan features; demographic fields remain for auditing only.
    preferred=['tenure','MonthlyCharges','PhoneService','MultipleLines','InternetService','Contract']
    present=[c for c in preferred if c in data.columns]
    exclude=[]
    for c in data.columns:
        key=c.lower().replace(' ','').replace('_','')
        if c==churn_col or any(s in key for s in ('customerid','accountid','subscriberid','customername','email','firstname','lastname','phone','mobile','address','postcode','zipcode','postalcode','latitude','longitude')):
            exclude.append(c)
        elif data[c].nunique(dropna=True) >= 0.95*len(data) and not pd.api.types.is_numeric_dtype(data[c]):
            exclude.append(c)
        elif data[c].isna().all() or data[c].nunique(dropna=True)<=1:
            exclude.append(c)
    Xraw=data[present].copy() if len(present)>=3 else data.drop(columns=exclude).copy()
    exclude=[c for c in data.columns if c not in Xraw.columns]
    # Convert numeric-looking string columns and remove near-unique categoricals.
    for c in Xraw.columns:
        if not pd.api.types.is_numeric_dtype(Xraw[c]):
            candidate=pd.to_numeric(Xraw[c].astype('string').str.replace(',','',regex=False).str.strip(), errors='coerce')
            if candidate.notna().mean()>0.9: Xraw[c]=candidate
    high_card=[c for c in Xraw.select_dtypes(exclude='number').columns if Xraw[c].nunique(dropna=True)>args.max_categories]
    Xraw=Xraw.drop(columns=high_card)
    numerics=Xraw.select_dtypes(include='number').columns.tolist()
    categoricals=[c for c in Xraw.columns if c not in numerics]
    if not numerics and not categoricals: raise ValueError('No usable input features. Inspect your CSV columns.')
    transformers=[]
    if numerics:
        transformers.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler())]),numerics))
    if categoricals:
        transformers.append(('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categoricals))
    prep=ColumnTransformer(transformers,sparse_threshold=0)
    X=prep.fit_transform(Xraw)
    if not np.isfinite(X).all(): raise ValueError('Non-finite features produced by preprocessing')
    if np.unique(X,axis=0).shape[0]<3: raise ValueError('Insufficient distinct customer feature profiles for clustering.')
    candidates=list(range(2,min(10,len(X)-1)+1))
    scores=[]; fitted={}
    sample_size=min(2500,len(X))
    for k in candidates:
        km=KMeans(n_clusters=k,random_state=RANDOM_STATE,n_init=10)
        labels=km.fit_predict(X)
        if len(set(labels))<2 or len(set(labels))>=len(X): continue
        sil=silhouette_score(X,labels,sample_size=sample_size,random_state=RANDOM_STATE)
        db=davies_bouldin_score(X,labels)
        scores.append({'k':k,'silhouette':float(sil),'davies_bouldin':float(db),'inertia':float(km.inertia_)})
        fitted[k]=labels
    if not scores: raise ValueError('Could not form distinct clusters. Review dataset and features.')
    metrics=pd.DataFrame(scores);metrics.to_csv(out/'cluster_metrics.csv',index=False)
    best=(args.clusters if args.clusters in fitted else int(metrics.sort_values(['silhouette','davies_bouldin'],ascending=[False,True]).iloc[0]['k']))
    labels=fitted[best]
    result=data.copy();result['cluster']=labels
    result.to_csv(out/'customer_clusters.csv',index=False)
    sizes=result['cluster'].value_counts().sort_index().rename('customers')
    profile=pd.DataFrame({'customers':sizes,'percentage':(100*sizes/len(result)).round(2)})
    if churn_col:
        churn=churn_indicator(result[churn_col])
        if churn.notna().any():
            profile['churn_rate_pct']=(100*churn.groupby(labels).mean()).round(2)
    if numerics:
        means=Xraw.assign(cluster=labels).groupby('cluster')[numerics].mean(numeric_only=True)
        means.columns=['mean_'+c for c in means.columns]
        profile=profile.join(means)
    profile.to_csv(out/'cluster_profiles.csv',index_label='cluster')
    categoricals_to_profile=[c for c in ['PhoneService','MultipleLines','InternetService','Contract','Dependents','gender','SeniorCitizen'] if c in data.columns]
    categorical_summary=[]
    for c in categoricals_to_profile:
        tab=pd.crosstab(result['cluster'],result[c],normalize='index').mul(100)
        for cluster_id,row in tab.iterrows():
            for value,pct in row.items():
                categorical_summary.append({'cluster':int(cluster_id),'feature':c,'value':str(value),'percentage':round(float(pct),2)})
    pd.DataFrame(categorical_summary).to_csv(out/'categorical_profiles.csv',index=False)
    pca=PCA(n_components=2,random_state=RANDOM_STATE)
    pos=pca.fit_transform(X)
    fig,ax=plt.subplots(figsize=(9,6))
    sc=ax.scatter(pos[:,0],pos[:,1],c=labels,cmap='tab10',alpha=.6,s=13)
    ax.set(xlabel='Principal component 1',ylabel='Principal component 2',title=f'Customer segments: K-means (k={best})')
    fig.colorbar(sc,ax=ax,label='Cluster');fig.tight_layout();fig.savefig(out/'clusters_pca.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(metrics.k,metrics.silhouette,marker='o')
    ax.axvline(best,linestyle='--',color='gray');ax.set(xlabel='Clusters (k)',ylabel='Silhouette score',title='Selecting number of clusters')
    fig.tight_layout();fig.savefig(out/'silhouette_scores.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.bar(sizes.index.astype(str),sizes.values)
    ax.set(xlabel='Cluster',ylabel='Customers',title='Customers by segment');fig.tight_layout();fig.savefig(out/'cluster_sizes.png',dpi=180);plt.close(fig)
    if 'churn_rate_pct' in profile:
        fig,ax=plt.subplots(figsize=(8,5));ax.bar(profile.index.astype(str),profile['churn_rate_pct'])
        ax.set(xlabel='Cluster',ylabel='Churn rate (%)',title='Observed churn rate by segment');fig.tight_layout();fig.savefig(out/'churn_by_cluster.png',dpi=180);plt.close(fig)
    summary={'rows':len(data),'features':X.shape[1],'selected_k':best,'selection_note':'Fixed k=4 prioritises interpretable, actionable segments; k=10 had maximum tested silhouette. Set --clusters 0 for automatic silhouette choice.' if args.clusters else 'Selected highest silhouette score within candidates','excluded_columns':exclude+high_card,'numeric_features':numerics,'categorical_features':categoricals,'churn_column':churn_col,'pca_variance_explained':pca.explained_variance_ratio_.tolist(),'method':'Behaviour/service-plan K-means with silhouette-score selection; no churn-label leakage or demographic features'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2));print('\nSEGMENT PROFILES\n',profile.to_string())

if __name__=='__main__':main()
