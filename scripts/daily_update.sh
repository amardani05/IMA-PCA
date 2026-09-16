#!/bin/zsh
# Unattended daily refresh for the IMA risk screener / factor library.
#
# Run by launchd every weekday morning (see scripts/com.ima.pca.daily.plist,
# installed to ~/Library/LaunchAgents). Mirrors the Sell-Model daily refresh:
#   1. full pipeline run (fresh price/fundamentals caches)
#   2. on success: commit the regenerated webapp data (data files only,
#      never source code), best-effort push, deploy to Vercel production
#   3. on failure: retry once after a cool-down (wake-time network is the
#      usual culprit), then give up; the site keeps yesterday's data and a
#      macOS notification says so — a silent failure is how the 8/25 and
#      8/31–9/16 outages went unnoticed.
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

# Per-attempt cap. A normal run is ~4 min on cached data; the weekly
# fundamentals refetch on a slow wake-time network was 55 min on 9/16. 150 min
# leaves headroom for that while still catching a true hang (8/25: 5 days).
PIPELINE_LIMIT_S=9000
DEPLOY_LIMIT_S=600
RETRY_DELAY_S=1200

notify() {
  # Native macOS banner so a failed refresh is visible without opening logs.
  /usr/bin/osascript -e "display notification \"$1\" with title \"IMA daily refresh\"" \
    >/dev/null 2>&1 || true
}

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "$(date) another refresh is already running; exiting" >> "$LOG"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

cd "$REPO" || exit 1
echo "=== daily refresh started $(date) ===" >> "$LOG"

# Network preflight: launchd fires on wake before WiFi/DNS are up, and a
# pipeline started without network fetches a gutted universe (see the
# Sell-Model 2026-07-20 incident). Require the three upstreams the pipeline
# depends on to all answer; wait up to 10 minutes; abort otherwise.
net_ready() {
  for url in "https://query2.finance.yahoo.com/" \
             "https://api.stlouisfed.org/" \
             "https://www.sec.gov/"; do
    /usr/bin/curl -s -o /dev/null --max-time 8 "$url" || return 1
  done
  return 0
}
NET_OK=0
for i in $(seq 1 30); do
  if net_ready; then NET_OK=1; break; fi
  echo "network not ready (attempt $i/30); sleeping 20s" >> "$LOG"
  sleep 20
done
if [ "$NET_OK" -ne 1 ]; then
  echo "=== NO NETWORK after 10 minutes $(date); aborting, site keeps previous data ===" >> "$LOG"
  notify "No network after 10 min — refresh skipped, site keeps previous data"
  exit 1
fi

# Watchdog: run a command with a hard time limit. NOTE: zsh reserves
# `status` as a read-only alias of $? — assigning to it aborts the function
# (the 8/31–9/16 outage: every run "failed" on that line after a successful
# pipeline, so nothing was ever committed or deployed). Use `rc`.
run_with_timeout() {
  local limit_s="$1"; shift
  "$@" >> "$LOG" 2>&1 &
  local cmd_pid=$!
  ( sleep "$limit_s"; echo "WATCHDOG: killing PID $cmd_pid after ${limit_s}s" >> "$LOG"
    kill -9 "$cmd_pid" 2>/dev/null ) &
  local dog_pid=$!
  wait "$cmd_pid"
  local rc=$?
  kill "$dog_pid" 2>/dev/null
  return $rc
}

# Pipeline: one retry after a cool-down. The second attempt is cheap — every
# cache the first attempt managed to fill is reused — and it usually lands on
# a stable network instead of the just-woke-up one.
PIPELINE_OK=0
for attempt in 1 2; do
  echo "--- pipeline attempt $attempt $(date) ---" >> "$LOG"
  if run_with_timeout "$PIPELINE_LIMIT_S" "$PY" main.py; then
    PIPELINE_OK=1; break
  fi
  if [ "$attempt" -eq 1 ]; then
    echo "--- pipeline attempt 1 failed $(date); retrying in $((RETRY_DELAY_S/60)) min ---" >> "$LOG"
    sleep "$RETRY_DELAY_S"
  fi
done

if [ "$PIPELINE_OK" -ne 1 ]; then
  echo "=== PIPELINE FAILED after 2 attempts $(date); site keeps previous data ===" >> "$LOG"
  notify "Pipeline failed twice — site keeps previous data. See output/daily/"
  exit 1
fi

echo "--- pipeline OK, committing data $(date) ---" >> "$LOG"
git add webapp/public data/factors_manual >> "$LOG" 2>&1
if ! git diff --cached --quiet; then
  git commit -m "Daily data refresh $(date +%Y-%m-%d)" >> "$LOG" 2>&1
  git push origin main >> "$LOG" 2>&1 \
    || echo "push failed (credentials?); commit stays local" >> "$LOG"
else
  echo "no data changes to commit" >> "$LOG"
fi

# Deploy from the repo root: the Vercel project's Root Directory setting is
# "webapp", so the CLI must upload the repo root (deploying from inside
# webapp/ fails with "Root Directory does not exist").
cd "$REPO" || exit 1
if run_with_timeout "$DEPLOY_LIMIT_S" "$VERCEL" --prod --yes; then
  echo "=== DEPLOYED OK $(date) ===" >> "$LOG"
else
  echo "=== DEPLOY FAILED $(date); data committed locally ===" >> "$LOG"
  notify "Vercel deploy failed — data committed locally, site not updated"
  exit 1
fi
