# Fix: Remove Large File from Git History

## Problem
GitHub rejected your push because `model-state.bin` (415.50 MB) exceeds the 100 MB file size limit.

## Solution: Remove from Git History

### Option 1: Quick Fix (Recommended)

Run these commands in your terminal:

```bash
# 1. Remove the file from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch model-state.bin" \
  --prune-empty --tag-name-filter cat -- --all

# 2. Clean up git references
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 3. Verify the file is removed
git log --all --name-only | grep model-state.bin
# Should return nothing

# 4. Push again (you may need to force push if you've already pushed)
git push --force-with-lease origin main
```

### Option 2: Using the Script

```bash
# Run the provided script
bash scripts/utils/remove_large_file.sh

# Then push
git push --force-with-lease origin main
```

## Important Notes

⚠️ **Warning**: This rewrites git history. If others are working on this repo:
- Coordinate with your team first
- Use `--force-with-lease` instead of `--force` (safer)
- Everyone will need to re-clone or reset their local repos

## Verify It's Ignored

The file is already in `.gitignore`, so it won't be tracked in the future:

```bash
# Check .gitignore
cat .gitignore | grep model-state
# Should show: model-state.bin
```

## Alternative: Use Git LFS (for future)

If you need to store large model files in git, consider using Git LFS:

```bash
# Install git-lfs (if not installed)
# macOS: brew install git-lfs
# Then:
git lfs install
git lfs track "*.bin"
git add .gitattributes
git commit -m "Add Git LFS tracking for .bin files"
```

But for now, just keep model files local and don't commit them.
