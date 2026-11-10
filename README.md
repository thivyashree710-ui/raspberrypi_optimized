# Raspberry Pi 4 - vehicle detection, counting and web dashboard (optimized)

Live vehicle detection and counting for a traffic camera on a **Raspberry Pi 4 (4 x Cortex-A72, no accelerator)**, using an **INT8 YOLO11s model (18 Indian road-vehicle classes) run with ExecuTorch + XNNPACK**.
It reads the camera over RTSP, counts each vehicle once when it crosses a counting line, and serves a web dashboard (live video, reports, recordings) in the same layout as the Jetson Xavier NX dashboard.

## What it does
- Detection on the Pi: INT8 model, 4 threads, runs in its own process; every frame is detected (`SYNC` mode) or the picture is smoothed between detections (`SMOOTH_25FPS`).
- **Counting line** (drawn with the mouse) and **ignore-zones** (areas where detections are dropped).
- **Tracker**: one ID per vehicle, class chosen by voting over all its detections, one count per vehicle. A dead-band around the line and a memory for stationary vehicles mean a vehicle parked on the line for many minutes is counted once, even if its detection flickers.
- **Web dashboard** (`--web`): Live Monitor (live video, latest counted vehicles with photos), Reports (live / hourly / daily / monthly, Excel export, filters), Recordings (start/stop, H.264 playback, delete).
- **Auto-start service** (systemd) for the Pi.
- Camera credentials are never stored in the code and never printed.

## Measured on the Pi 4 (live camera, SYNC)
| Input | Model call | Pipeline | Notes |
|---|---|---|---|
| 256 x 256 | 58-66 ms | about 13 FPS | best accuracy; detection rate limited to about 14-15 per second |
| 128 x 128 | about 27-35 ms | about 24-25 FPS | finds fewer small/far vehicles than the 256 model |

A window or a VNC viewer costs CPU; the Pi has no fan and throttles near 80 C. Decoding the camera **main** stream (1920x1080) costs much more CPU than the **sub** stream (704x576) and gives no accuracy benefit, because the picture is scaled to 704x576 before detection: use the sub stream (`STREAM=sub`) for long runs.

## Requirements
Python 3.13 with `executorch 1.5.1`, `torch 2.15.0+cpu`, `opencv-python 5.0.0`, `numpy` (a `.pte` file is tied to the ExecuTorch version). The web part uses only the standard library plus OpenCV/numpy.

**The model files are not in this repository.** Put `yolo_256_int8_xnnpack.pte` (and `yolo_128_int8_xnnpack.pte`) next to `New_optimized.py`. `tools/export_fast.py` re-exports a checkpoint (fewer XNNPACK graph breaks, about 13 % faster at 256).

## Camera address
Never put it in the code. Either
```
export RTSP_URL='rtsp://user:password@camera-ip:554/sub'
```
or a private file `.rtsp.env` next to `New_optimized.py` containing `RTSP_URL=rtsp://user:password@camera-ip:554/sub` (it is in `.gitignore`).

## Run
```
python New_optimized.py --weights yolo_256_int8_xnnpack.pte                       # window on the Pi screen, every frame detected
python New_optimized.py --weights yolo_128_int8_xnnpack.pte --imgsz 128 --web     # web dashboard on port 8000
python New_optimized.py --draw-line      # draw / edit the counting line (saved to count_line.json)
python New_optimized.py --draw-zones     # draw / edit the ignore-zones (saved to exclude_zones.json)
```
Useful options: `--mode SYNC_DETECTION|SMOOTH_25FPS`, `--threads N`, `--workers N`, `--source file.mp4`, `--loop`, `--no-gui`, `--save-video out.mp4`, `--track-log`, `--line-margin`, `--park-memory-s`, `--web-port`, `--web-host`, `--rec-dir`, `--web-data`. Run with `-h` for all.

## Web dashboard
`http://<pi-address>:8000` : `/` Live Monitor, `/recordings`, `/custom` (Reports), `/hourly`, `/daily`, `/monthly`.
**There is no login**: anyone who can reach the port can open the pages and start/stop/delete recordings. Use a trusted network or `--web-host 127.0.0.1`.
Counted vehicles are stored in SQLite (`web_data/traffic.db`) with a photo of each one, so reports survive restarts.

## Auto-start on boot
Edit `service/web.env` (model, stream `main`/`sub`, port, folders), then on the Pi: `bash service/install_service.sh` (asks for sudo). Remove with `service/uninstall_service.sh`. Paths in the scripts assume `/mnt/raspi-ssd/New`; keep recordings, data and logs on the SSD, not the SD card.

## Layout
```
New_optimized.py     detector, tracker, counting line, zones, windows, web hand-over
web_ui.py            web dashboard (own process) + web_assets/ (stylesheet, logo, fonts)
service/             systemd unit, start script, settings, install/uninstall
tools/export_fast.py model re-export
tests/               developer scripts: synthetic tracker tests (synth.py), replay of recorded detections
docs/                earlier notes and the model export configuration
```
`tests/` scripts contain local paths from the development PC; edit them before use. `python tests/synth.py new` runs the tracker checks (parked vehicle with dropouts, side-by-side vehicles, reversing, class flicker).

## Credits
The dashboard layout and stylesheet follow the JETSON_XAVIER_NX_MIC710AIX project.
