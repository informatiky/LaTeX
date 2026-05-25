#!/bin/bash
# Configuration
REPO_PATH="$HOME/Duzzle/LaTeX"   # Change to your VS Code project folder
COMMIT_INTERVAL=30
PUSH_INTERVAL=600

# Trim logs older than 1 day
find "$(dirname "$0")" -name "*.log" -mtime +1 -delete

cd "$REPO_PATH" || exit 1
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)

pull_changes() {
    cd "$REPO_PATH" || exit 1
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    BEFORE_HASH=$(git rev-parse HEAD)
    git pull origin "$BRANCH_NAME" >/dev/null 2>&1
    AFTER_HASH=$(git rev-parse HEAD)
    if [ "$BEFORE_HASH" = "$AFTER_HASH" ]; then
        echo "[$TIMESTAMP] Already up to date."
    else
        echo "[$TIMESTAMP] Pulled changes from remote."
    fi
}

commit_changes() {
    cd "$REPO_PATH" || exit 1
    if [[ -n $(git status --porcelain) ]]; then
        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
        git add --all
        git commit -m "Auto-commit: $TIMESTAMP"
        echo "[$TIMESTAMP] Changes committed locally."
        return 0
    else
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] No changes detected."
        return 1
    fi
}

push_changes() {
    cd "$REPO_PATH" || exit 1
    UNPUSHED=$(git log --branches --not --remotes --oneline | wc -l)
    if [ "$UNPUSHED" -gt 0 ]; then
        TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
        git push origin "$BRANCH_NAME"
        echo "[$TIMESTAMP] Pushed $UNPUSHED commit(s) to remote."
    fi
}

echo "Starting auto-commit for $REPO_PATH on branch $BRANCH_NAME"
echo "Commits every ${COMMIT_INTERVAL}s, pushes every $((PUSH_INTERVAL / 60))min"

COUNTER=0
PUSH_COUNT=$((PUSH_INTERVAL / COMMIT_INTERVAL))

while true; do
    commit_changes
    COUNTER=$((COUNTER + 1))
    if [ $COUNTER -ge $PUSH_COUNT ]; then
        push_changes
        COUNTER=0
    fi
    sleep "$COMMIT_INTERVAL"
done