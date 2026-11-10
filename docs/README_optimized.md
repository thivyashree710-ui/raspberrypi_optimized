# New_optimized.py - how to run it (directly on the Raspberry Pi, with the window)

A separate file next to `New.py` (untouched). Same model (`yolo_256_int8_xnnpack.pte`), same outputs (counts CSV, sample images, benchmark report), written to `hackathon_submission_optimized/` so your existing `hackathon_submission/` is never overwritten.

## 1. Run it on the Pi screen (GUI window)
The window needs the OpenCV build that has a GUI: **use the `venv_gui` Python** (the normal `venv` has a headless OpenCV and cannot open a window).
```
cd /mnt/raspi-ssd/New
export RTSP_URL="$(grep -o 'rtsp://[^"]*' New.py | head -1)"      # camera sub-stream address taken from your New.py; it is never printed (password masked)
/mnt/raspi-ssd/projects/mobilenet_pi4/venv_gui/bin/python New_optimized.py --mode ASYNC_STREAM_25FPS --workers 1 --threads 3     # smooth ~25 FPS picture
/mnt/raspi-ssd/projects/mobilenet_pi4/venv_gui/bin/python New_optimized.py                                                      # SYNC: boxes locked to every frame
```
Press `q` or `Esc` in the window to stop; the CSV, sample images and report are written when it ends.
Use the sub-stream (`/sub`, 704x576). The main stream is a much larger picture and is very expensive to decode on a Pi 4.
Other options: `--no-gui`, `--workers N`, `--threads N`, `--agnostic-nms` (old New.py behaviour), `--letterbox`, `--loop` (restart a video file), `--max-frames N`, `--source file.mp4`.

## 2. Measured on the Pi 4 (INT8 model, 256x256)
| Test | New.py | New_optimized.py |
|---|---|---|
| Video file, SYNC, no window | 73.1 ms/frame, 13.4-13.6 FPS | 70.9-71.9 ms/frame, 13.4-13.7 FPS |
| Video file, ASYNC display, no window | 13.7 FPS (the model call blocks the display thread) | 24.5-24.8 FPS (detections 11.6-13.2 per second) |
| Live camera /sub, SYNC, no window | - | 12.1 FPS (80.9 ms per frame) |
| Live camera /sub, ASYNC, no window | - | 24.6 FPS display, 10.4 detections per second |
| Live camera + window on the Pi screen, ASYNC | - | **24.5 FPS display, 7.8 detections per second** |
| Live camera + window on the Pi screen, SYNC | - | 9.1 FPS |
Drawing the window costs about 30 % of the detection speed in SYNC mode; ASYNC keeps the picture smooth because the model runs in its own process.
Other results: 2 worker processes give only +5-7 % in SYNC mode (14.5 FPS, 127 ms delay); 3 or 4 workers are slower. The model call scales with threads (193 / 110 / 80 / 67 ms for 1 / 2 / 3 / 4), so keep 4 threads in SYNC mode.
Detection counts are identical to New.py with `--agnostic-nms`; class-aware NMS (default) adds about 0.5 % detections on the test clip.

## 3. What changed compared with New.py
- The camera password is not in the code and is never printed (it was hard-coded and printed in five places). Consider changing the camera password, since the old file displayed it.
- Class-aware NMS by default (the old class-agnostic NMS can remove a rider / bike / person that overlap).
- ASYNC mode now uses a worker process: ExecuTorch holds Python's GIL, so a thread could not run the detector and keep a 25 FPS display.
- Explicit ExecuTorch thread count, reusable buffers, in-place drawing, a camera reader that keeps only the newest frame without blocking, reconnect if the stream stalls, measured throughput reporting.

## 4. Known limits
Tested on a video file and on the live camera sub-stream with and without the window. Not tested: other cameras, runs longer than about a minute, and the `--letterbox` option on the camera.

## 5. How these numbers were measured (methodology)
- **Device:** Raspberry Pi 4 Model B Rev 1.4 (4x Cortex-A72 at 1.8 GHz, 7.6 GB RAM, governor `ondemand`), no fan. Python 3.13, ExecuTorch 1.5.1, torch 2.15.0+cpu, OpenCV 5.0.0.
- **Model:** `yolo_256_int8_xnnpack.pte` (INT8, XNNPACK, input 1x3x256x256, 6 raw outputs; see `model_export_config.md`). Confidence 0.35, NMS IoU 0.45, frames stretched (not letterboxed) to 256x256, as in New.py.
- **Inputs:** (a) a 10 s traffic clip, 640x360, 319 frames; (b) the same clip re-encoded to 704x576 (the camera sub-stream size) for the window tests; (c) the live camera sub-stream (704x576), 200-300 frames per run.
- **Baseline:** New.py logic run from a sanitized copy (camera address removed, GUI switched off, otherwise identical) in a separate scratch folder, so its outputs never overwrite `hackathon_submission/`.
- **Conditions:** the Pi was cooled to about 62 C or below before every run (waiting up to 3 minutes); no other program was running; each run started a fresh process (model load and 3 warm-up calls are not timed). Runs ended at 70-77 C; the soft-temperature-limit flag (`0x80000`) was already set from earlier use, and no run showed active throttling at its end. Repeated runs of the same setting agreed within about 1 % (13.60 / 13.62 / 13.68 FPS).
- **What each number means:**
  - *per-frame latency* = pre-processing + model call + decode/NMS, measured around those steps in the process that runs the model;
  - *measured throughput / display FPS* = (frames shown - 1) / (time of the last shown frame - time of the first), so start-up and shutdown are NOT counted (an earlier version counted worker shutdown and under-reported; fixed and re-measured);
  - *detections per second* (ASYNC) = detector results completed / the same wall time. In SYNC mode it equals the display FPS.
- **Thread scaling and GIL tests:** the model call alone was timed for 1-4 threads with the ExecuTorch thread pool set explicitly (60 calls each); the GIL test ran a 1 ms ticker thread next to repeated model calls and recorded its longest gap (128 ms with `portable_lib`, 204 ms with `Runtime.execute`).
- **Not covered:** runs longer than about a minute, other cameras, accuracy against ground-truth labels (only agreement with the original model was checked for the NCNN comparison, not for this INT8 model), and a Pi 5.
