#!/bin/zsh
# Unattended daily refresh for the IMA torpedo screener.
#
# Run by launchd every weekday morning (see scripts/com.ima.pca.daily.plist,
# installed to ~/Library/LaunchAgents). Mirrors the Sell-Model daily refresh:
#   1. full pipeline run (fresh price/fundamentals caches)
#   2. on success: commit the regenerated webapp data (data files only,
#      never source code), best-effort push, deploy to Vercel production
#   3. on failure: log and stop; the site keeps yesterday's data
#
# Logged to output/daily/refresh_YYYYMMDD_HHMM.log. A lock directory prevents
# overlapping runs (e.g. a laptop waking twice).

set -u
REPO="/Users/amardani/IMA-PCA"
PY="/Users/amardani/anaconda3/bin/python3"
VERCEL="/Users/amardani/.npm-global/bin/vercel"
LOG_DIR="$REPO/output/daily"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/refresh_$(date +%Y%m%d_%H%M).log"
LOCK="/tmp/ima_pca_daily.lock"

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date) another refresh is already running; exiting" >> "$LOG"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

cd "$REPO" || exit 1
echo "=== daily refresh started $(date) ===" >> "$LOG"

# Network preflight: launchd fires on wake before WiFi/DNS are up, and a
# pipeline started without network fetches a gutted universe (see the
# Sell-Model 2026-07-20 incident). Wait up to 10 minutes; abort otherwise.
NET_OK=0
for i in $(seq 1 30); do
  if /usr/bin/curl -s --max-time 5 "https://query2.finance.yahoo.com/" >/dev/null 2>&1; then
    NET_OK=1; break
  fi
  echo "network not ready (attempt $i/30); sleeping 20s" >> "$LOG"
  sleep 20
done
if [ "$NET_OK" -ne 1 ]; then
  echo "=== NO NETWORK after 10 minutes $(date); aborting, site keeps previous data ===" >> "$LOG"
  exit 1
fi

if "$PY" main.py >> "$LOG" 2>&1; then
  echo "--- pipeline OK, committing data $(date) ---" >> "$LOG"
  git add webapp/public >> "$LOG" 2>&1
  if ! git diff --cached --quiet -- webapp/public; then
    git commit -m "Daily data refresh $(date +%Y-%m-%d)" -- webapp/public >> "$LOG" 2>&1
    git push origin main >> "$LOG" 2>&1 \
      || echo "push failed (credentials?); commit stays local" >> "$LOG"
  else
    echo "no data changes to commit" >> "$LOG"
  fi
  # Deploy from the repo root: the Vercel project's Root Directory setting is
  # "webapp", so the CLI must upload the repo root (deploying from inside
  # webapp/ fails with "Root Directory does not exist").
  cd "$REPO" || exit 1
  if "$VERCEL" --prod --yes >> "$LOG" 2>&1; then
    echo "=== DEPLOYED OK $(date) ===" >> "$LOG"
  else
    echo "=== DEPLOY FAILED $(date); data committed locally ===" >> "$LOG"
    exit 1
  fi
else
  echo "=== PIPELINE FAILED $(date); site keeps previous data ===" >> "$LOG"
  exit 1
fi
