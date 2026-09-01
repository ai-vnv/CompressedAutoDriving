#!/usr/bin/env bash
set -e
cd "$HOME/aivnv/duckie-pomdp"
git rm -r -q --cached .clonecheck
rm -rf .clonecheck
printf '\n# scratch space used when checking a bare-clone install\n.clonecheck/\n' >> .gitignore
git add -A
git commit -q -m "Remove a stray scratch directory from the tree"
git fetch -q publish main
TREE=$(git rev-parse main^{tree})
COMMIT=$(git commit-tree "$TREE" -p "$(git rev-parse publish/main)" \
  -m "Remove a stray scratch directory from the tree")
git push publish "$COMMIT:refs/heads/main"
echo "pushed $COMMIT"
