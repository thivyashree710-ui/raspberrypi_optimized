# Model file and export configuration

## The model file
| | |
|---|---|
| File name | `yolo_256_int8_xnnpack.pte` |
| Size | 9,769,032 bytes (9.32 MiB) |
| SHA-256 | `cc114cd9237e075864ffcaa543b7aee5cbb1c33a05a443277d7f75f254beb477` |
| Format / runtime | ExecuTorch `.pte`, XNNPACK delegate, INT8. Run with ExecuTorch 1.5.1 and torch 2.15.0+cpu on the Raspberry Pi 4 (`.pte` files are tied to the ExecuTorch version) |
| Made by | `model_converter_code.py` (included; the camera address in it is masked) |

## Which checkpoint it was exported from (not proven)
The saved export script does **not** record the source weights: its `MODEL_PATH` variable currently holds the `.pte` file name (left over from editing) and a comment mentions `yolov8n.pt`. The 18-class checkpoint stored next to the model is `18cls_9L_last_recovered.pt` (SHA-256 `ac3213698a100f4aec2b7f4e9d6dba462002efcc784737346dbd663d86da5ce5`, a custom YOLO11s with 18 classes, 9.43 M parameters, trained at 640).
The `.pte` has 18-channel class outputs and, on traffic video, detects the 18-class names correctly (auto_rickshaw, two_Wheeler, car, lcv, ...), so it is almost certainly that 18-class model, but I have not verified byte-for-byte which checkpoint produced it.

## Input
`1 x 3 x 256 x 256`, float32, RGB, values 0..1 (divide by 255). The camera frame is **stretched** (plain resize, not letterboxed) to 256 x 256; the calibration images were preprocessed the same way.

## Outputs (6 tensors, float32, raw: no box decoding, no sigmoid and no NMS inside the model)
| Output | Shape | Meaning |
|---|---|---|
| 0, 1, 2 | `[1,64,32,32]`, `[1,64,16,16]`, `[1,64,8,8]` | box distribution logits: 4 sides x 16 bins, strides 8 / 16 / 32 |
| 3, 4, 5 | `[1,18,32,32]`, `[1,18,16,16]`, `[1,18,8,8]` | class logits (apply sigmoid) for the 18 classes |

Decoding used in `New.py` and `New_optimized.py`: confidence threshold 0.35 (applied on the logit), softmax over the 16 bins and expectation to get the four distances, grid-cell centre +0.5, multiply by the stride, scale to the camera picture, NMS IoU 0.45 (`New_optimized.py`: class-aware by default).

## Classes (id: name)
0 person, 1 two_Wheeler, 2 auto_rickshaw, 3 car, 4 lcv, 5 Eicher, 6 van, 7 mini_bus, 8 bus, 9 Truck_2Axle, 10 Truck_3Axle, 11 Truck_4Axle, 12 Truck_6Axle, 13 multi_axle, 14 others_earth_mover, 15 tracktor, 16 tracktor_trailer, 17 cycle.

## Export configuration (from `model_converter_code.py`)
- **Graph change 1:** every `SiLU` activation is replaced by `x * sigmoid(x)` (`XNNPACKSiLU`), so both operations quantize natively in XNNPACK INT8 and do not fall back to FP32.
- **Graph change 2:** the Detect head is split into its raw convolution branches (`cv2[0..2]` = box, `cv3[0..2]` = class) and returned directly; decoding is done outside the model.
- **Quantization:** PyTorch 2 export (PT2E) static INT8 with `XNNPACKQuantizer`, `set_global(get_symmetric_quantization_config(is_per_channel=True))` (symmetric, per-channel weights), `prepare_pt2e` -> calibration -> `convert_pt2e`.
- **Calibration data:** up to 50 images from the folder `calib_images` (if fewer than 5 are present the script captures 30 frames from the camera, every 5th frame); each is resized (stretched) to 256 x 256, BGR to RGB, divided by 255. With no images it falls back to 16 random tensors. The number of images actually used for this `.pte` is not recorded.
- **Lowering:** `torch.export.export(..., strict=False)` then `to_edge_transform_and_lower` with `XnnpackPartitioner()`, `transform_passes=get_transform_passes()`, `EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)`. The script prints the number of XNNPACK delegate partitions (target: 1); the value for this file was not recorded.

## Measured on the Raspberry Pi 4 with this file
Model call 66.5 ms with 4 threads (193 / 110 / 80 / 67 ms for 1 / 2 / 3 / 4 threads); about 71-73 ms per frame including pre-processing and decoding; 12.1 FPS (SYNC) and 24.6 FPS display (ASYNC) on the live camera sub-stream. Details and methodology: `README_optimized.md`.

## Not known / not measured
Accuracy of this INT8 model against ground-truth labels; the exact calibration set; which checkpoint it came from (see above); the delegate partition count.
