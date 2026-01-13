# Setting Up Your Own Git Repository

## Step 1: Remove the existing git repository

```bash
cd "/Users/dinalbandara/Desktop/IIT/4th year/FYP/sample-models/Resume-NER"
rm -rf .git
```

## Step 2: Initialize a new git repository

```bash
git init
```

## Step 3: Add all files to git

```bash
git add .
```

## Step 4: Make your first commit

```bash
git commit -m "Initial commit: Resume NER project with trained model"
```

## Step 5: Add your remote repository

If you have a GitHub/GitLab/Bitbucket repository already created:

```bash
git remote add origin <your-repo-url>
# Example: git remote add origin https://github.com/yourusername/resume-ner.git
```

## Step 6: Push to your repository

```bash
git branch -M main  # Rename branch to main (if needed)
git push -u origin main
```

## Important Notes:

1. **Model File Size**: The `model-state.bin` file is 415MB. 
   - If you want to include it in git, it's fine (but will make your repo large)
   - If you want to exclude it, uncomment the `# model-state.bin` line in `.gitignore` before `git add .`

2. **Virtual Environment**: The `venv/` folder is already in `.gitignore` (you don't want to commit this)

3. **Check what will be committed**:
   ```bash
   git status
   ```

## Alternative: If you want to exclude the model file

If the model file is too large for your git repo, you can exclude it:

1. Edit `.gitignore` and uncomment: `model-state.bin`
2. Then follow steps 2-6 above

The model file will stay on your local machine, but won't be tracked by git.
