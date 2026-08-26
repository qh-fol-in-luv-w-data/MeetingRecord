#!/usr/bin/env bash
# Creates a GitHub repo and pushes this project to it.
#
# Requirements:
#   - GitHub CLI installed and authenticated: https://cli.github.com
#     Check with: gh auth status
#     Login with: gh auth login
#   - git installed
#
# Usage:
#   ./create_and_push_repo.sh <repo-name> [public|private]
#
# Example:
#   ./create_and_push_repo.sh vi-diar-asr private

set -euo pipefail

REPO_NAME="${1:-MeetingRecorder}"
VISIBILITY="${2:-private}"

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "Second argument must be 'public' or 'private' (got: $VISIBILITY)"
  exit 1
fi

if ! command -v gh &> /dev/null; then
  echo "GitHub CLI (gh) not found. Install it first:"
  echo "  macOS:  brew install gh"
  echo "  Linux:  see https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
  exit 1
fi

if ! gh auth status &> /dev/null; then
  echo "Not logged in to GitHub CLI. Run 'gh auth login' first, then re-run this script."
  exit 1
fi

echo "== Initializing git repo (if not already) =="
if [ ! -d .git ]; then
  git init
  git branch -M main
fi

echo "== Staging and committing files =="
git add .
if git diff --cached --quiet; then
  echo "Nothing to commit (working tree already matches last commit)."
else
  git commit -m "Initial commit: vi-diar-asr diarization + Qwen3-ASR LoRA finetuning pipeline"
fi

echo "== Creating GitHub repo '$REPO_NAME' ($VISIBILITY) and pushing =="
gh repo create "$REPO_NAME" \
  --"$VISIBILITY" \
  --source=. \
  --remote=origin \
  --push

echo "== Done =="
gh repo view --web || true
