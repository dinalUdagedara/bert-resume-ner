#!/bin/bash
# Script to remove model-state.bin from git history

echo "Removing model-state.bin from git history..."

# Remove from all commits using git filter-branch
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch model-state.bin" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "✅ Done! The file has been removed from git history."
echo "⚠️  Note: This rewrites history. If you've already pushed, you'll need to force push."
echo "   Use: git push --force-with-lease origin main"
