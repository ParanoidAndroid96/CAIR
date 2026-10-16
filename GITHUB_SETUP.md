# GitHub setup

From the directory containing this repository:

```bash
cd CAIR
git init
git add .
git commit -m "Initial CAIR release"
git branch -M main
```

Create an empty repository on GitHub, then connect and push it:

```bash
git remote add origin https://github.com/<USERNAME>/<REPOSITORY>.git
git push -u origin main
```

If you use SSH:

```bash
git remote add origin git@github.com:<USERNAME>/<REPOSITORY>.git
git push -u origin main
```

Recommended repository name:

```text
CAIR
```

Before making the repository public, add the software license you want to use through GitHub or as a `LICENSE` file.
