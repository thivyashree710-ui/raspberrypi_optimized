#!/bin/bash
# Starts the detector + website. Run by systemd (new-optimized-web.service); can also be run by hand for a test.
# The camera address comes from the RTSP_URL environment variable, or /mnt/raspi-ssd/New/.rtsp.env (private file, not in the repository), and is handed to the program in an environment variable only: it is never printed.
ENV_FILE="${ENV_FILE:-/mnt/raspi-ssd/New/service/web.env}"
cd /mnt/raspi-ssd/New || exit 1
# wait (up to 10 minutes) until the SSD is mounted and writable - never write to the SD card
ok=0; for i in $(seq 1 120); do if mountpoint -q /mnt/raspi-ssd && touch /mnt/raspi-ssd/New/.service_write_test 2>/dev/null; then rm -f /mnt/raspi-ssd/New/.service_write_test; ok=1; break; fi; sleep 5; done
[ "$ok" = 1 ] || { echo "SSD /mnt/raspi-ssd is not mounted/writable - not starting"; exit 1; }
. "$ENV_FILE"
export AUTO_RECORD RECORD_KEEP_HOURS RECORD_MIN_FREE_GB RECORD_BLOCK_MINUTES RECORD_SEG_SECONDS          # hourly video + Excel report (see web.env)
LOG="${LOG:-/mnt/raspi-ssd/New/service/web.log}"; mkdir -p "$(dirname "$LOG")" "$WEB_DATA" "$REC_DIR"
[ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 5242880 ] && mv -f "$LOG" "$LOG.1"
U="${RTSP_URL:-}"
[ -n "$U" ] || { [ -f /mnt/raspi-ssd/New/.rtsp.env ] && U="$(grep '^RTSP_URL=' /mnt/raspi-ssd/New/.rtsp.env | head -1 | cut -d= -f2- | tr -d "'\"")"; }
[ -n "$U" ] || U="$(grep -o 'rtsp://[^"]*' /mnt/raspi-ssd/New/New.py 2>/dev/null | head -1)"
[ -n "$U" ] || { echo "no camera address: put  RTSP_URL=rtsp://user:password@camera:554/sub  into /mnt/raspi-ssd/New/.rtsp.env"; exit 1; }
case "$U" in                                                      # STREAM=main / sub picks the camera's stream (Hikvision-style .../Streaming/Channels/101 = main, 102 = sub; or the older .../main and .../sub)
  */Streaming/Channels/101|*/Streaming/Channels/102) base="${U%/*}"; if [ "$STREAM" = "main" ]; then U="$base/101"; else U="$base/102"; fi;;
  *) if [ "$STREAM" = "main" ]; then case "$U" in */sub) U="${U%/sub}/main";; esac; fi;;
esac
export RTSP_URL="$U" PYTHONDONTWRITEBYTECODE=1
exec /mnt/raspi-ssd/projects/mobilenet_pi4/venv/bin/python /mnt/raspi-ssd/New/New_optimized.py \
  --weights "$MODEL" --imgsz "$IMGSZ" --mode SYNC_DETECTION --threads "$THREADS" \
  ${REFINE_MODEL:+--refine-model "$REFINE_MODEL"} \
  ${TRACK_LOG:+--track-log} \
  --web --web-port "$PORT" --web-host "$HOST" --web-data "$WEB_DATA" --rec-dir "$REC_DIR" \
  --out /mnt/raspi-ssd/New/service/out --print-every 100000 >> "$LOG" 2>&1
