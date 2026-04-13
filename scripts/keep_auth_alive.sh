#!/bin/bash
# Periodically refresh gcloud application default credentials to prevent token expiry.
# Usage: ./scripts/keep_auth_alive.sh [interval_minutes]
# Default interval: 45 minutes (tokens expire after 60)

INTERVAL_MIN=${1:-45}
INTERVAL_SEC=$((INTERVAL_MIN * 60))

echo "Refreshing credentials every ${INTERVAL_MIN} minutes (Ctrl+C to stop)"

while true; do
    TOKEN=$(gcloud auth application-default print-access-token 2>&1)
    if [ $? -eq 0 ]; then
        echo "[$(date)] Token refreshed successfully (${TOKEN:0:20}...)"
    else
        echo "[$(date)] ERROR: Failed to refresh token: $TOKEN"
        echo "Run: gcloud auth application-default login"
    fi
    sleep "$INTERVAL_SEC"
done
