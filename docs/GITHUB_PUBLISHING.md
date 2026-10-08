# Publishing to GitHub

Recommended repository name:

`computer-vision-fundamentals-lab`

Recommended one-line description:

`From-scratch NumPy implementations and experiments for digital image processing and computer vision fundamentals.`

Suggested topics:

`computer-vision`, `image-processing`, `numpy`, `hough-transform`, `canny-edge-detection`, `non-local-means`, `steganography`, `interpolation`, `psnr`, `mse`

## Option A - GitHub CLI

From the repository root:

```bash
git init
git add .
git commit -m "Initial portfolio release: assignments 01 and 02"
git branch -M main
gh repo create computer-vision-fundamentals-lab --public --source=. --remote=origin --push
```

If Git asks for your identity first:

```bash
git config user.name "YOUR NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

## Option B - GitHub website

1. Create a new empty public repository named `computer-vision-fundamentals-lab`.
2. Do not add a README, `.gitignore`, or license on the website; this folder already contains them where needed.
3. Copy the repository URL, then run:

```bash
git init
git add .
git commit -m "Initial portfolio release: assignments 01 and 02"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

## Before making it public

- Review `docs/ASSET_NOTICE.md` and confirm the redistribution status of the included input images.
- Open the root README on GitHub and confirm that all result figures render correctly.
- Confirm that no student ID, private submission report, or other personal data has been added.
- Check the Actions tab after the first push; the validation workflow should run automatically.

## Adding the next assignment

Create `assignments/assignment-03/`, follow `docs/ADDING_ASSIGNMENTS.md`, update the root README, and add the new tests to `scripts/validate_repository.py`.
