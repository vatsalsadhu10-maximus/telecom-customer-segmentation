# Publish to GitHub

1. Sign in to GitHub and select **New repository**.
2. Repository name: `telecom-customer-segmentation`.
3. Choose **Public** only if the code is approved for sharing; leave **Add a README** unchecked.
4. Select **Create repository**.
5. On your computer, extract this ZIP and open a terminal inside the extracted folder.
6. Run:

```bash
git init
git add .
git commit -m "Add reproducible telecom customer clustering analysis"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/telecom-customer-segmentation.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username and authenticate when prompted. The `.gitignore` excludes the private input data and customer-level cluster output. To upload using GitHub's web UI instead, choose **Add file → Upload files** in your new repository and drag in the project files/folders (including `src/`). Check privacy before publishing anything else.
