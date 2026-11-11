#!/usr/bin/env python3
"""New_optimized.py - ExecuTorch INT8 traffic detection + counting for the Raspberry Pi 4 (18 custom classes).  Same model, same outputs as New.py; this is a separate file (New.py is untouched).

WHAT CHANGED (measured on the Pi 4, see README_optimized.md):
  * Camera address is NOT in the source any more and is never printed (password masked).  Provide it with the RTSP_URL environment variable or a private file `.rtsp.env` next to this script.
  * NMS is class-aware (like Ultralytics) by default; New.py used class-agnostic NMS, which could suppress a rider / a bike / a person that overlap.   --agnostic-nms restores the old behaviour.
  * Parallel worker processes (--workers N): ExecuTorch holds Python's GIL, so threads can not overlap inference with anything; separate processes can.  N=1 is the original single-process path.
  * Thread pool of ExecuTorch/XNNPACK is set explicitly (--threads); OMP_NUM_THREADS / torch.set_num_threads do not control it.
  * Reusable buffers for pre-processing, in-place drawing, HUD darkened only where it is drawn, camera thread keeps only the newest frame (no lock held while waiting) and reconnects if the stream drops, console printed every few frames.
  * --no-gui works (the old --gui flag could not be switched off), --letterbox optional, throughput reported as measured frames/s (not 1000/latency).
Run:  python New_optimized.py --source video.mp4 --weights yolo_256_int8_xnnpack.pte --workers 2 --no-gui       (RTSP: export RTSP_URL=... first, or put RTSP_URL=... in .rtsp.env, chmod 600)"""
import math, os, sys, re, csv, glob, time, argparse, threading, socket
import multiprocessing as mp
from collections import Counter, deque

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|skip_loop_filter;all|threads;2")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false;*.debug=false"); os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8"); os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
if sys.platform != "win32" and not os.environ.get("DISPLAY"): os.environ["DISPLAY"] = ":0"
import numpy as np, cv2
cv2.setNumThreads(1)                                    # OpenCV threads must not fight the model threads

# ------------------------------------------------------------------ configuration (same names/defaults as New.py where they exist)
MODEL_PATH = "yolo_256_int8_xnnpack.pte"; RUN_MODE = "SYNC_DETECTION"; TARGET_FPS = 25.0; IMG_SIZE = 256; CONF_THRESHOLD = 0.35; IOU_THRESHOLD = 0.45
SHOW_GUI = True; DISPLAY_WIDTH = 704; DISPLAY_HEIGHT = 576; AUTO_SUBSTREAM = True; OUTPUT_DIR = "hackathon_submission_optimized"; MAX_SAMPLE_SAVES = 5
TRAFFIC_CLASSES = ["person", "two_Wheeler", "auto_rickshaw", "car", "lcv", "Eicher", "van", "mini_bus", "bus", "Truck_2Axle", "Truck_3Axle", "Truck_4Axle", "Truck_6Axle", "multi_axle", "others_earth_mover", "tracktor", "tracktor_trailer", "cycle"]
STRIDES = (8, 16, 32); DFL_BINS = np.arange(16, dtype=np.float32)


# ------------------------------------------------------------------ credentials: never hard-coded, never printed
def load_rtsp_url():
    u = os.environ.get("RTSP_URL", "").strip()
    if u: return u
    p = os.path.join(HERE, ".rtsp.env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line.startswith("RTSP_URL="): return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def mask_url(u): return re.sub(r"(rtsp://[^:/@]+:)[^@]*@", r"\1***@", str(u))


# ------------------------------------------------------------------ pre-processing (reusable buffers)
class PreProc:
    def __init__(self, S, letterbox=False):
        self.S, self.letterbox = S, letterbox; self.small = np.empty((S, S, 3), np.uint8); self.rgb = np.empty((S, S, 3), np.uint8); self.canvas = np.full((S, S, 3), 114, np.uint8)

    def __call__(self, frame, out):
        """frame BGR -> RGB 0..1 CHW float32 written into `out` (1,3,S,S). Returns (inv_rx, inv_ry, padx, pady): net px -> original px = (x - pad) * inv_r."""
        S = self.S; h, w = frame.shape[:2]
        if not self.letterbox:
            cv2.resize(frame, (S, S), dst=self.small, interpolation=cv2.INTER_LINEAR); src = self.small; geo = (w / S, h / S, 0.0, 0.0)
        else:
            r = min(S / h, S / w); nw, nh = int(round(w * r)), int(round(h * r)); px, py = (S - nw) // 2, (S - nh) // 2
            self.canvas[:] = 114; self.canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR); src = self.canvas; geo = (1 / r, 1 / r, float(px), float(py))
        cv2.cvtColor(src, cv2.COLOR_BGR2RGB, dst=self.rgb); np.multiply(self.rgb.transpose(2, 0, 1), 1.0 / 255.0, out=out[0], casting="unsafe")
        return geo


# ------------------------------------------------------------------ decoding of the 6 raw heads (3 x box DFL logits, 3 x class logits)
PERSON_CLASS = 0
OCC_AGE, OCC_COVER = 45, 0.30                                                  # a vehicle that vanished BEHIND another detection is kept (predicted) for up to 45 frames (~2 s) instead of 8; 'behind' = at least this share of its box is covered
MOVE_EFF, MOVE_TOTAL, MOVE_MIN_DET = 0.60, 0.60, 6                      # a long track must really MOVE (one direction, or far) to be counted: parked vehicle with a fidgeting rider is not traffic
REFINE_CONF = 0.60                                                             # second look only below this live confidence
SHAPE_WIDE, SHAPE_CONF, SHAPE_ALT = 1.0, 0.5, 0.25                      # shape check thresholds (box width/height, mean confidence, share of the other class's votes)
REFINE_WEIGHT = 4.0                 # weight of the zoom re-classification in the class vote (a track's detections add up to roughly 3-8)
CLUSTER_IOU = 0.55                  # candidates that overlap this much are the SAME object, whatever class they carry
CLUSTER_POWER = 2.0                 # class evidence = sum of score**power over all candidates of that class in the cluster (confident candidates count more)


def cluster_nms(B, SC, CI, iou_thr=CLUSTER_IOU, power=CLUSTER_POWER):
    """Class-AGNOSTIC non-maximum suppression with class voting. The old class-aware NMS kept one box PER CLASS on the same object (e.g. 'auto_rickshaw' 0.56 and 'lcv' 0.35 on one rickshaw),
    which the tracker then counted twice. Here every group of candidates that overlap (IoU > iou_thr) becomes ONE detection: the box of the strongest candidate, the class with the largest
    combined evidence, and the best score of that class."""
    n = len(SC)
    if n == 0: return [], [], []
    b = np.asarray(B, np.float32); x1, y1 = b[:, 0], b[:, 1]; x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]; area = b[:, 2] * b[:, 3]; order = np.argsort(-np.asarray(SC)); used = np.zeros(n, bool); ob, os_, oc = [], [], []
    for i in order:
        if used[i]: continue
        iw = np.maximum(0.0, np.minimum(x2[i], x2) - np.maximum(x1[i], x1)); ih = np.maximum(0.0, np.minimum(y2[i], y2) - np.maximum(y1[i], y1)); inter = iw * ih; iou = inter / (area[i] + area - inter + 1e-9)
        mem = (~used) & (iou > iou_thr); mem[i] = True; ev = {}; best = {}
        for j in np.nonzero(mem)[0]:
            c = int(CI[j]); ev[c] = ev.get(c, 0.0) + float(SC[j]) ** power; best[c] = max(best.get(c, 0.0), float(SC[j]))
        cls = max(ev, key=ev.get); used |= mem; ob.append([int(v) for v in B[i]]); os_.append(best[cls]); oc.append(cls)
    return ob, os_, oc


def _decode_raw(outs, orig_w, orig_h, S, conf, iou, geo, agnostic=False, cluster=True, drop_person=True):
    """outs: 6 numpy arrays [(1,64,h,w) x3 box, (1,18,h,w) x3 class]. Same maths as New.py (sub-pixel rounding, clipping); class-aware NMS unless agnostic."""
    conf_c = min(max(conf, 1e-4), 0.9999); thr = -np.log(1.0 / conf_c - 1.0); irx, iry, padx, pady = geo
    B, SC, CI = [], [], []
    for i, stride in enumerate(STRIDES):
        c = outs[3 + i][0]; ml = c.max(0); gy, gx = np.nonzero(ml >= thr)
        if gy.size == 0: continue
        sc = 1.0 / (1.0 + np.exp(-ml[gy, gx])); cid = c[:, gy, gx].argmax(0)
        b = outs[i][0][:, gy, gx].T.reshape(-1, 4, 16); e = np.exp(b - b.max(2, keepdims=True)); d = ((e / e.sum(2, keepdims=True)) * DFL_BINS).sum(2)
        ax = gx.astype(np.float32) + 0.5; ay = gy.astype(np.float32) + 0.5
        x1n = (ax - d[:, 0]) * stride; y1n = (ay - d[:, 1]) * stride; bwn = (d[:, 0] + d[:, 2]) * stride; bhn = (d[:, 1] + d[:, 3]) * stride
        x1 = np.round((x1n - padx) * irx).astype(np.int32); y1 = np.round((y1n - pady) * iry).astype(np.int32); bw = np.round(bwn * irx).astype(np.int32); bh = np.round(bhn * iry).astype(np.int32)
        x1 = np.clip(x1, 0, orig_w - 1); y1 = np.clip(y1, 0, orig_h - 1); bw = np.clip(bw, 1, orig_w - x1); bh = np.clip(bh, 1, orig_h - y1)
        B.append(np.column_stack((x1, y1, bw, bh))); SC.append(sc.astype(np.float32)); CI.append(cid.astype(np.int32))
    if not B: return [], [], []
    B = np.concatenate(B); SC = np.concatenate(SC); CI = np.concatenate(CI)
    if drop_person: keep_ = CI != PERSON_CLASS; B, SC, CI = B[keep_], SC[keep_], CI[keep_]                  # a pedestrian or a rider is not a counted vehicle
    if len(SC) == 0: return [], [], []
    if cluster and not agnostic: return cluster_nms(B.tolist(), SC.tolist(), CI.tolist())
    bl, sl = B.tolist(), SC.tolist()
    idx = cv2.dnn.NMSBoxes(bl, sl, conf, iou) if agnostic else cv2.dnn.NMSBoxesBatched(bl, sl, CI.tolist(), conf, iou)
    if len(idx) == 0: return [], [], []
    idx = np.array(idx).flatten(); return [bl[i] for i in idx], [sl[i] for i in idx], [int(CI[i]) for i in idx]


# ------------------------------------------------------------------ exclusion zones (areas where detections are ignored)
def load_zones(path):
    """JSON {"zones": [[[x, y], ...], ...]} with x, y as fractions of the picture width / height (0..1), so the zones work at any resolution."""
    import json
    with open(path, encoding="utf-8") as fh: z = json.load(fh).get("zones", [])
    return [np.asarray(q, np.float32) for q in z if len(q) >= 3]


_ZCACHE = {}
def zones_px(zones, w, h):
    k = (id(zones), w, h)
    if k not in _ZCACHE: _ZCACHE[k] = [np.round(q * np.array([w, h], np.float32)).astype(np.int32) for q in zones]
    return _ZCACHE[k]


def decode_heads(outs, orig_w, orig_h, S, conf, iou, geo, agnostic=False, zones=None, zpoint="center", cluster=True, drop_person=True):
    """Decodes the 6 raw outputs, then drops every detection whose reference point (box centre, or bottom-centre of the box) lies inside an exclusion zone."""
    b, sc, ci = _decode_raw(outs, orig_w, orig_h, S, conf, iou, geo, agnostic, cluster, drop_person)
    if not zones or not b: return b, sc, ci
    polys = zones_px(zones, orig_w, orig_h); keep = []
    for i, (x, y, w, h) in enumerate(b):
        pt = (float(x + w / 2), float(y + (h if zpoint == "bottom" else h / 2)))
        if not any(cv2.pointPolygonTest(q, pt, False) >= 0 for q in polys): keep.append(i)
    return [b[i] for i in keep], [sc[i] for i in keep], [ci[i] for i in keep]


def _near_vertex(polys, x, y, r=12):
    """(zone index, corner index) of the saved corner closest to (x, y) within r pixels, else None."""
    best, hit = r * r + 1, None
    for pi, q in enumerate(polys):
        for vi, (vx, vy) in enumerate(q):
            d = (vx - x) ** 2 + (vy - y) ** 2
            if d < best: best, hit = d, (pi, vi)
    return hit


def draw_zones_tool(frame, out_path):
    """Edit the ignore-zones. Your saved zones are loaded. Left-drag a corner = move it | left-click on empty picture = add a corner of a new zone | Enter or right-click = close the new zone |
    right-click INSIDE a zone = delete that zone | u = undo last new corner | c = remove all zones | s = save and quit | q/Esc = quit without saving."""
    import json
    h, w = frame.shape[:2]; cur = []; win = "Edit ignore-zones"; drag = [None]
    polys = [[(int(round(x * w)), int(round(y * h))) for x, y in q] for q in (load_zones(out_path) if os.path.exists(out_path) else [])]
    print(f"loaded {len(polys)} saved zone(s) from {out_path}" if polys else "no saved zones yet - draw new ones")
    def mouse(ev, x, y, fl, pr):
        if ev == cv2.EVENT_LBUTTONDOWN:
            hit = _near_vertex(polys, x, y)
            if hit: drag[0] = hit
            else: cur.append((x, y))
        elif ev == cv2.EVENT_MOUSEMOVE and drag[0]: polys[drag[0][0]][drag[0][1]] = (min(max(x, 0), w - 1), min(max(y, 0), h - 1))
        elif ev == cv2.EVENT_LBUTTONUP: drag[0] = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            if len(cur) >= 3: polys.append(list(cur)); cur.clear()
            else:
                for i in range(len(polys) - 1, -1, -1):
                    if cv2.pointPolygonTest(np.array(polys[i], np.int32), (float(x), float(y)), False) >= 0: del polys[i]; break
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, min(w, 960), int(min(w, 960) * h / w)); cv2.setMouseCallback(win, mouse)
    while True:
        img = frame.copy(); ov = img.copy()
        for q in polys: cv2.fillPoly(ov, [np.array(q, np.int32)], (0, 0, 255))
        img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        for q in polys:
            cv2.polylines(img, [np.array(q, np.int32)], True, (0, 0, 255), 2); [cv2.circle(img, c_, 5, (255, 255, 255), -1) for c_ in q]
        if cur: cv2.polylines(img, [np.array(cur, np.int32)], False, (0, 255, 255), 2); [cv2.circle(img, c_, 4, (0, 255, 255), -1) for c_ in cur]
        cv2.putText(img, "drag corner: move | click: new corner | Enter/right-click: close | right-click inside zone: delete | u undo | c clear | s save | q quit", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img); k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and len(cur) >= 3: polys.append(list(cur)); cur.clear()
        elif k == ord("u") and cur: cur.pop()
        elif k == ord("c"): polys.clear(); cur.clear()
        elif k == ord("s"):
            if len(cur) >= 3: polys.append(list(cur))
            json.dump({"zones": [[[round(x / w, 4), round(y / h, 4)] for x, y in q] for q in polys], "reference_size": [w, h]}, open(out_path, "w"), indent=1); print(f"saved {len(polys)} zone(s) to {out_path}"); break
        elif k in (ord("q"), 27): print("quit without saving"); break
    cv2.destroyAllWindows()


# ------------------------------------------------------------------ vehicle tracking + counting line
def _iou_c(ax, ay, aw, ah, bx, by, bw, bh):
    iw = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2); ih = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    if iw <= 0 or ih <= 0: return 0.0
    i = iw * ih; return i / (aw * ah + bw * bh - i + 1e-9)


class VehicleTracker:
    """Keeps ONE id per vehicle across detections and counts it ONCE when it crosses the counting line.
    - matching: best overlap with the predicted position; a candidate of a very different size, another vehicle group (two-wheeler / auto-rickshaw / car-truck) or the opposite
      direction of travel is penalised, so neighbouring vehicles do not swap ids;
    - crossing: a dead-band of `margin` (fraction of the picture height) on both sides of the line - a vehicle counts only after it was CLEARLY on one side and is then CLEARLY on the
      other, so a stopped vehicle whose detection box jitters across the line is never counted;
    - one count per vehicle: a vehicle that was counted and is lost while standing still is remembered for `park_s` seconds; a new track that appears on the same spot takes over its id
      and its 'already counted' state, so a vehicle parked on the line for many minutes (even if its detection flickers) is shown only once;
    - class: the class with the highest summed confidence over all its detections."""
    def __init__(self, line, flip=False, point="center", min_hits=3, max_age=8, margin=0.025, park_s=1800.0, groups=None):
        self.line, self.flip, self.point, self.min_hits, self.max_age, self.margin, self.park_s, self.groups = line, flip, point, min_hits, max_age, margin, park_s, (groups or {})
        self.tracks = []; self.next_id = 1; self.done = []; self.info = {}; self.mem = []; self.recent = deque(); self.refiner = None; self.rstat = {"checked": 0, "answered": 0, "changed": 0}; self.counts = {"IN": Counter(), "OUT": Counter()}

    def _grp(self, c): return self.groups.get(c, c)

    @staticmethod
    def _radius(t):
        hs = t["hist"]; mx = sum(p[0] for p in hs) / len(hs); my = sum(p[1] for p in hs) / len(hs); return max(math.hypot(p[0] - mx, p[1] - my) for p in hs)

    @staticmethod
    def _mean(t): hs = t["hist"]; return sum(p[0] for p in hs) / len(hs), sum(p[1] for p in hs) / len(hs)

    @staticmethod
    def _drift(t):
        """How far the vehicle has really travelled over the last detections: distance between the average position of the first and of the last third (random jitter of the box averages out)."""
        hs = list(t["hist"]); k = max(1, len(hs) // 3); a = hs[:k]; b = hs[-k:]
        return math.hypot(sum(p[0] for p in a) / len(a) - sum(p[0] for p in b) / len(b), sum(p[1] for p in a) / len(a) - sum(p[1] for p in b) / len(b))

    def _real_move(self, t):
        """True if the vehicle is really travelling: net distance / path length over its last detections is high (real traffic is almost 1.0; a parked scooter whose box jitters back and forth is low), or it has travelled well over its own size since it appeared. Short tracks are not judged."""
        h = t["hist"]
        if len(h) < MOVE_MIN_DET: return True
        size = max(t["w"], t["h"], 1.0); p0 = t.get("p0") or h[0]
        if math.hypot(h[-1][0] - p0[0], h[-1][1] - p0[1]) >= MOVE_TOTAL * size: return True
        path = sum(math.hypot(h[i + 1][0] - h[i][0], h[i + 1][1] - h[i][1]) for i in range(len(h) - 1)); net = math.hypot(h[-1][0] - h[0][0], h[-1][1] - h[0][1])
        return net >= MOVE_EFF * max(path, 1e-6)

    def _stationary(self, t): return len(t["hist"]) >= 8 and self._drift(t) <= max(14.0, 0.15 * max(t["w"], t["h"])) and self._radius(t) <= 0.6 * max(t["w"], t["h"])          # sits on one spot (box jitter does not matter)
    def _moving(self, t): return len(t["hist"]) >= 6 and (self._drift(t) > 0.25 * max(t["w"], t["h"]) or self._radius(t) > 0.8 * max(t["w"], t["h"]))
    def totals(self): return sum(self.counts["IN"].values()), sum(self.counts["OUT"].values())

    @staticmethod
    def _geom(A, B, p):
        dx, dy = B[0] - A[0], B[1] - A[1]; L2 = (dx * dx + dy * dy) or 1.0
        return (dx * (p[1] - A[1]) - dy * (p[0] - A[0])) / math.sqrt(L2), ((p[0] - A[0]) * dx + (p[1] - A[1]) * dy) / L2          # signed distance to the line (px), position along the segment (0..1)

    def _state(self, d, m): return 1 if d > m else (-1 if d < -m else 0)

    def _side_update(self, t, p, A, B, m):
        d, u = self._geom(A, B, p); st = self._state(d, m)
        if st == 0: return
        if t["side"] is None: t["side"], t["side_pt"], t["side_d"] = st, p, d; return
        if st != t["side"] and t["cross"] is None:
            p0, d0 = t["side_pt"], t["side_d"]; f = d0 / (d0 - d); q = (p0[0] + f * (p[0] - p0[0]), p0[1] + f * (p[1] - p0[1])); _, uq = self._geom(A, B, q)
            if -0.02 <= uq <= 1.02: t["cross"] = "IN" if ((t["side"] < 0) != self.flip) else "OUT"
            else: t["note"] = "went round the end of the counting line"
        t["side"], t["side_pt"], t["side_d"] = st, p, d

    def update(self, boxes, scores, cids, w, h, fid, now=None):
        now = time.monotonic() if now is None else now; A = (self.line[0][0] * w, self.line[0][1] * h); B = (self.line[1][0] * w, self.line[1][1] * h); m = self.margin * h; det = []
        for (x, y, bw, bh), sc, c in zip(boxes, scores, cids): det.append((x + bw / 2.0, y + bh / 2.0, float(bw), float(bh), float(sc), int(c), (x + bw / 2.0, y + (bh if self.point == "bottom" else bh / 2.0)), self._vote_w(x, y, bw, bh, w, h)))
        self.mem = [e for e in self.mem if now - e["t"] <= self.park_s]; pairs = []
        for ti, t in enumerate(self.tracks):
            px, py = self._mean(t) if (t["miss"] > 0 and self._stationary(t)) else (t["cx"] + t["vx"], t["cy"] + t["vy"])
            for di, d in enumerate(det):
                iou = _iou_c(px, py, t["w"], t["h"], d[0], d[1], d[2], d[3])
                if iou >= 0.1: q = iou + 0.5
                else:
                    dist = math.hypot(px - d[0], py - d[1]); lim = 1.5 * max(t["w"], t["h"], 1.0); q = 0.4 * (1 - dist / lim) if dist < lim else 0.0
                    if q > 0 and math.hypot(t["vx"], t["vy"]) > 2.0 and (t["vx"] * (d[0] - t["cx"]) + t["vy"] * (d[1] - t["cy"])) < 0: q *= 0.6        # moving the opposite way to this track
                ratio = (d[2] * d[3]) / max(t["w"] * t["h"], 1.0)
                if q > 0 and iou < 0.5 and not (0.35 <= ratio <= 2.8): q = 0.0                                                                            # very different size: another vehicle
                if q > 0: q *= min(ratio, 1.0 / max(ratio, 1e-6))                                                                                          # prefer the detection of the SAME size (a small vehicle inside a big one's box must not take its id)
                if q > 0 and self._grp(d[5]) != self._grp(t["lastc"]): q *= 0.7
                if q > 0: pairs.append((q, ti, di))
        pairs.sort(reverse=True); ut, ud, ids, events = set(), set(), [0] * len(det), []
        for q, ti, di in pairs:
            if ti in ut or di in ud: continue
            ut.add(ti); ud.add(di); t = self.tracks[ti]; d = det[di]; t["vx"] = 0.5 * t["vx"] + 0.5 * (d[0] - t["cx"]); t["vy"] = 0.5 * t["vy"] + 0.5 * (d[1] - t["cy"])
            t["cx"], t["cy"], t["w"], t["h"] = d[0], d[1], d[2], d[3]; t["hits"] += 1; t["miss"] = 0; t["occ"] = False; t["lastc"] = d[5]; t["votes"][d[5]] = t["votes"].get(d[5], 0.0) + d[4] * d[7]; t["sums"][d[5]] = t["sums"].get(d[5], 0.0) + d[4]; t["pt"] = d[6]; ids[di] = t["id"]; t["last"] = fid; t["hist"].append((d[0], d[1]))
            self._side_update(t, d[6], A, B, m)
            if t["cross"] and not t["counted"] and t["hits"] >= self.min_hits and self._real_move(t): self._count(t, fid, events)                   # (a vehicle that is not really moving is held back, and counted as soon as it is)
        for t in self.tracks:
            if self.mem and t["miss"] == 0 and self._moving(t): self.mem = [e for e in self.mem if _iou_c(e["cx"], e["cy"], e["w"], e["h"], t["cx"], t["cy"], t["w"], t["h"]) <= 0.55 or (t["w"] * t["h"]) < 0.5 * (e["w"] * e["h"])]      # a moving vehicle of about the same size went right through that spot: whatever was parked there is gone (a small scooter passing in front of a standing car does not erase it)
        for ti, t in enumerate(self.tracks):
            if ti not in ut:
                t["miss"] += 1
                if not self._stationary(t): t["cx"] += t["vx"]; t["cy"] += t["vy"]
                if not t.get("occ") and t["hits"] >= self.min_hits:                                         # hidden behind another vehicle? (its last / predicted box is mostly covered by a detection)
                    ta = max(t["w"] * t["h"], 1.0)
                    for d in det:
                        ix = min(t["cx"] + t["w"] / 2, d[0] + d[2] / 2) - max(t["cx"] - t["w"] / 2, d[0] - d[2] / 2); iy = min(t["cy"] + t["h"] / 2, d[1] + d[3] / 2) - max(t["cy"] - t["h"] / 2, d[1] - d[3] / 2)
                        if ix > 0 and iy > 0 and ix * iy / ta >= OCC_COVER and d[2] * d[3] >= 0.8 * ta: t["occ"] = True; break
        n_old = len(self.tracks)
        for di, d in enumerate(det):
            if di in ud: continue
            sl = None                                                                                   # a small piece of a vehicle that is hidden behind another one (same kind of vehicle, inside its predicted box): part of THAT vehicle, not a new one
            for ti in range(n_old):
                o = self.tracks[ti]
                if ti in ut or not o.get("occ") or self._grp(d[5]) != self._grp(o["lastc"]): continue
                ix = min(o["cx"] + o["w"] / 2, d[0] + d[2] / 2) - max(o["cx"] - o["w"] / 2, d[0] - d[2] / 2); iy = min(o["cy"] + o["h"] / 2, d[1] + d[3] / 2) - max(o["cy"] - o["h"] / 2, d[1] - d[3] / 2)
                if ix > 0 and iy > 0 and ix * iy >= 0.8 * d[2] * d[3] and d[2] * d[3] <= 0.5 * o["w"] * o["h"]: sl = o; break
            if sl is not None: ids[di] = sl["id"]; ud.add(di); continue
            dup = next((mi for mi in ud if _iou_c(d[0], d[1], d[2], d[3], det[mi][0], det[mi][1], det[mi][2], det[mi][3]) > 0.5), None)
            if dup is not None: ids[di] = ids[dup]; continue                                          # that object already has a track: no second track (no second count)
            tr = {"id": 0, "cx": d[0], "cy": d[1], "w": d[2], "h": d[3], "vx": 0.0, "vy": 0.0, "hits": 1, "miss": 0, "votes": {d[5]: d[4] * d[7]}, "sums": {d[5]: d[4]}, "lastc": d[5], "pt": d[6], "cross": None, "counted": False, "first": fid, "last": fid, "side": None, "side_pt": d[6], "side_d": 0.0, "note": "", "hist": deque([(d[0], d[1])], maxlen=20), "p0": (d[0], d[1])}
            for e in self.mem:                                                                          # a stationary, already-counted vehicle that was lost for a moment and is seen again
                io_ = _iou_c(e["cx"], e["cy"], e["w"], e["h"], d[0], d[1], d[2], d[3])
                if (io_ > 0.5 or (self._grp(d[5]) == self._grp(e["lastc"]) and (io_ > 0.4 or math.hypot(d[0] - e["cx"], d[1] - e["cy"]) < 0.5 * max(e["w"], e["h"])))):
                    tr.update(id=e["id"], cross=e["cross"], counted=e["counted"], votes=dict(e["votes"]), sums=dict(e["sums"]), hits=e["hits"] + 1, first=e["first"], side=e["side"], side_pt=e["side_pt"], side_d=e["side_d"], note="same stationary vehicle seen again after a gap: not counted twice" if e["counted"] else ""); self.mem.remove(e); break
            if not tr["id"]: tr["id"] = self.next_id; self.next_id += 1
            self._side_update(tr, d[6], A, B, m); self.tracks.append(tr); ids[di] = tr["id"]
        alive = []
        for t in self.tracks:
            if t["miss"] <= (OCC_AGE if t.get("occ") else self.max_age): alive.append(t)
            else: self._close(t, fid, events, now)
        self.tracks = alive; return ids, events

    @staticmethod
    def _vote_w(x, y, bw, bh, W, H):
        """How much one detection counts in the class vote: large (close, clear) views count fully, small distant ones less; a box cut by the picture edge (partly visible) counts much less."""
        sw = max(0.25, min(1.0, math.sqrt((bw * bh) / float(W * H) / 0.04))); edge = x <= 2 or y <= 2 or x + bw >= W - 2 or y + bh >= H - 2
        return sw * (0.4 if edge else 1.0)

    def _count(self, t, fid, events):
        """Count a vehicle once - unless another track was counted a moment ago on the same spot (the same vehicle seen twice)."""
        box = (t["cx"], t["cy"], t["w"], t["h"])
        while self.recent and fid - self.recent[0][0] > 6: self.recent.popleft()                    # only the last ~0.25 s: a different vehicle cannot be on the same spot that soon, but the same object can have two tracks at once
        for f0, bx, tid0 in self.recent:
            if tid0 != t["id"] and _iou_c(box[0], box[1], box[2], box[3], bx[0], bx[1], bx[2], bx[3]) > 0.45:
                t["counted"] = True; t["note"] = "same vehicle as #%d: not counted twice" % tid0; return False
        lb = (t["hist"][-1][0], t["hist"][-1][1], t["w"], t["h"]) if t.get("hist") else box                # where the vehicle was LAST SEEN (the track position keeps moving on after the detections stop)
        for o in self.tracks:                                                                           # a stray second track on a vehicle that is already counted (short track, box nested in / overlapping the real one): not a new vehicle
            if o is not t and self._grp(o["lastc"]) == self._grp(t["lastc"]) and (o["counted"] or o["hits"] >= max(8, 3 * t["hits"])):             # a bigger / already counted track of the same kind of vehicle
                bx = (o["cx"], o["cy"], o["w"], o["h"]); ix = min(lb[0] + lb[2] / 2, bx[0] + bx[2] / 2) - max(lb[0] - lb[2] / 2, bx[0] - bx[2] / 2); iy = min(lb[1] + lb[3] / 2, bx[1] + bx[3] / 2) - max(lb[1] - lb[3] / 2, bx[1] - bx[3] / 2)
                inside = (ix * iy) / max(min(lb[2] * lb[3], bx[2] * bx[3]), 1.0) if ix > 0 and iy > 0 else 0.0
                if t["hits"] <= 8 and (_iou_c(lb[0], lb[1], lb[2], lb[3], bx[0], bx[1], bx[2], bx[3]) > 0.3 or (inside > 0.6 and t["hits"] <= 6)):             # (only short tracks: a long track that is partly hidden behind a counted vehicle is a vehicle of its own)
                    t["counted"] = True; t["note"] = "stray track on vehicle #%d: not counted twice" % o["id"]; return False
        for e in self.mem:
            if e["id"] != t["id"] and e["counted"] and (_iou_c(e["cx"], e["cy"], e["w"], e["h"], lb[0], lb[1], lb[2], lb[3]) > 0.4 or (self._grp(t["lastc"]) == self._grp(e["lastc"]) and math.hypot(lb[0] - e["cx"], lb[1] - e["cy"]) < 0.5 * max(e["w"], e["h"]))):
                t["counted"] = True; t["note"] = "same standing vehicle as #%d: not counted twice" % e["id"]; return False
        top_ = max(t["votes"], key=t["votes"].get)
        if self.refiner is not None and self._grp(top_) != 0 and t["sums"].get(top_, 0.0) / max(t["hits"], 1) < REFINE_CONF:     # only when the live model is unsure: all of its mistakes in the 108-photo test had confidence < 0.6        # second look with the 256 model for every counted vehicle except two-wheelers (the live model is already accurate there)
            try: r = self.refiner(t)
            except Exception: r = None
            self.rstat["checked"] += 1
            if r:
                before = max(t["votes"], key=t["votes"].get); t["votes"][r[0]] = max(t["votes"].values()) + 1e-3; t["sums"][r[0]] = max(t["sums"].get(r[0], 0.0), r[1] * max(t["hits"], 1))
                self.rstat["answered"] += 1; self.rstat["changed"] += int(max(t["votes"], key=t["votes"].get) != before)
        cls = max(t["votes"], key=t["votes"].get)
        auto_ = next((k for k, g in self.groups.items() if g == 1), None) if getattr(self, "groups", None) else None                   # shape check: an auto-rickshaw is TALL (width/height about 0.7); a clearly WIDE box called auto_rickshaw with low confidence
        if auto_ is not None and cls == auto_ and t["w"] / max(t["h"], 1.0) > SHAPE_WIDE and t["sums"].get(cls, 0.0) / max(t["hits"], 1) < SHAPE_CONF:     # is almost always an lcv / van / car: take the best OTHER class the tracker saw (if it saw one)
            alt = {k: v for k, v in t["votes"].items() if k != auto_ and self.groups.get(k) == 2}
            if alt and max(alt.values()) >= SHAPE_ALT * t["votes"][cls]: cls = max(alt, key=alt.get); self.rstat["shape"] = self.rstat.get("shape", 0) + 1
        t["cpos"] = (t["cx"], t["cy"], t["w"], t["h"]); t["counted"] = True; self.counts[t["cross"]][cls] += 1; events.append((t["id"], cls, t["cross"], fid)); self._note(t, cls); self.recent.append((fid, box, t["id"])); return True

    def _note(self, t, cls):
        self.info[t["id"]] = (round(min(1.0, t["sums"].get(cls, 0.0) / max(t["hits"], 1)), 3), (lambda c_: (int(c_[0] - t["w"] / 2), int(c_[1] - t["h"] / 2), int(t["w"]), int(t["h"])))(t["hist"][-1] if t.get("hist") else (t["cx"], t["cy"])))     # photo box = where the vehicle was last SEEN
        if len(self.info) > 400: self.info.pop(next(iter(self.info)))

    def _close(self, t, fid, events, now=0.0):
        """A track ends: count a crossing that was seen but not yet counted (needs >= 2 detections), remember a stationary crossed vehicle, and log why the vehicle was or was not counted."""
        if t["cross"] and not t["counted"] and t["hits"] >= 2 and not self._real_move(t): t["note"] = "crossed the line but never really moved (parked / box jitter): not counted"
        elif t["cross"] and not t["counted"] and t["hits"] >= 2:
            if self._count(t, fid, events): t["note"] = "counted when the track ended (crossed with only %d detections)" % t["hits"]
        stayed = bool(t["counted"] and t.get("cpos") and math.hypot(t["cx"] - t["cpos"][0], t["cy"] - t["cpos"][1]) < 0.8 * max(t["cpos"][2], t["cpos"][3]))      # counted and never really left the spot (standing in the queue, creeping, box jitter)
        if self._stationary(t) or stayed:
            mx, my = self._mean(t); self.mem.append({"id": t["id"], "cx": mx, "cy": my, "w": t["w"], "h": t["h"], "t": now, "counted": t["counted"], "cross": t["cross"], "votes": dict(t["votes"]), "sums": dict(t["sums"]), "lastc": t["lastc"], "first": t["first"], "hits": t["hits"], "side": t["side"], "side_pt": t["side_pt"], "side_d": t["side_d"]})
        if not t["note"]:
            if t["counted"]: t["note"] = "counted"
            elif t["cross"]: t["note"] = "crossed but only %d detection(s): too short to count" % t["hits"]
            else: t["note"] = "never clearly crossed the line"
        top = max(t["votes"], key=t["votes"].get) if t["votes"] else -1
        self.done.append((t["id"], t["first"], t["last"], t["hits"], top, round(t["votes"].get(top, 0.0) / max(t["hits"], 1), 2), t["cross"] or "", t["counted"], t["note"]))

    def flush(self, fid):
        ev = []
        for t in self.tracks: self._close(t, fid, ev)
        self.tracks = []; return ev


def load_flip(path):
    import json
    try:
        with open(path, encoding="utf-8") as fh: return bool(json.load(fh).get("flip", False))
    except Exception: return False


def load_line(path):
    import json
    with open(path, encoding="utf-8") as fh: q = json.load(fh).get("line")
    return [tuple(q[0]), tuple(q[1])] if q and len(q) == 2 else None


def draw_count_line(frame, line, flip):
    h, w = frame.shape[:2]; A = (int(line[0][0] * w), int(line[0][1] * h)); B = (int(line[1][0] * w), int(line[1][1] * h)); cv2.line(frame, A, B, (0, 255, 255), 2)
    dx, dy = B[0] - A[0], B[1] - A[1]; n = math.hypot(dx, dy) or 1.0; nx, ny = (-dy / n, dx / n); nx, ny = (-nx, -ny) if flip else (nx, ny); m = ((A[0] + B[0]) // 2, (A[1] + B[1]) // 2)       # arrow points to the IN side
    cv2.arrowedLine(frame, m, (int(m[0] + nx * 34), int(m[1] + ny * 34)), (0, 255, 255), 2, tipLength=0.4); cv2.putText(frame, "IN", (int(m[0] + nx * 40) - 8, int(m[1] + ny * 40) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_line_tool(frame, out_path, zones, flip):
    """Edit the counting line. A saved line is loaded: left-drag an end point to move it. With no line yet: click the 2 end points. r = redo from scratch | s = save and quit | q/Esc = quit without saving."""
    import json
    h, w = frame.shape[:2]; pts = []; win = "Edit counting line"; drag = [None]
    if os.path.exists(out_path):
        old = load_line(out_path)
        if old: pts = [(int(round(x * w)), int(round(y * h))) for x, y in old]; print(f"loaded the saved counting line from {out_path}")
    def mouse(ev, x, y, fl, pr):
        if ev == cv2.EVENT_LBUTTONDOWN:
            hit = _near_vertex([pts], x, y) if pts else None
            if hit: drag[0] = hit[1]
            elif len(pts) < 2: pts.append((x, y))
        elif ev == cv2.EVENT_MOUSEMOVE and drag[0] is not None: pts[drag[0]] = (min(max(x, 0), w - 1), min(max(y, 0), h - 1))
        elif ev == cv2.EVENT_LBUTTONUP: drag[0] = None
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, min(w, 960), int(min(w, 960) * h / w)); cv2.setMouseCallback(win, mouse)
    while True:
        img = frame.copy()
        if zones: cv2.polylines(img, zones_px(zones, w, h), True, (0, 0, 255), 1)
        if len(pts) == 2: draw_count_line(img, [(pts[0][0] / w, pts[0][1] / h), (pts[1][0] / w, pts[1][1] / h)], flip)
        for q in pts: cv2.circle(img, q, 6, (255, 255, 255), -1)
        cv2.putText(img, "drag an end point to move it | r: redo | s: save | q: quit  (arrow = IN direction)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img); k = cv2.waitKey(30) & 0xFF
        if k == ord("r"): pts.clear()
        elif k == ord("s") and len(pts) == 2:
            json.dump({"line": [[round(pts[0][0] / w, 4), round(pts[0][1] / h, 4)], [round(pts[1][0] / w, 4), round(pts[1][1] / h, 4)]], "reference_size": [w, h]}, open(out_path, "w"), indent=1); print(f"saved counting line to {out_path}"); break
        elif k in (ord("q"), 27): print("quit without saving"); break
    cv2.destroyAllWindows()


# ------------------------------------------------------------------ window in its OWN process (so the window size never slows detection / counting)
GUI_RING = 8
def gui_main(shm, shape, seq, stop, win, wsize, nice=15, core=None, gui_fps=0.0, steady=25.0, prescale=False):
    """Shows the finished pictures. The detector loop only copies a frame into shared memory and never waits for this process.
    steady > 0: frames are shown at a FIXED rate (steady per second) from a small buffer (~3 frames), so a slow frame in the detector does not make the picture stutter; gui_fps > 0 (without steady): redraw at most that often."""
    buf = np.frombuffer(shm, np.uint8).reshape(shape); last = 0; N = shape[0]; shown_t = []
    def show(img):
        """Scale to the CURRENT window size with cv2.resize (vectorised) and hand the window a picture that needs no further scaling; the window's own software scaling is much slower."""
        if not prescale: cv2.imshow(win, img); return                      # default: the window scales the picture itself and keeps its shape
        try:
            _, _, ww, hh = cv2.getWindowImageRect(win)
            if ww > 0 and hh > 0 and (ww, hh) != (img.shape[1], img.shape[0]): img = cv2.resize(img, (ww, hh), interpolation=cv2.INTER_LINEAR if ww * hh < 4 * img.shape[0] * img.shape[1] else cv2.INTER_NEAREST)
        except Exception: pass
        cv2.imshow(win, img)
    try: core is not None and os.sched_setaffinity(0, {core})
    except Exception: pass
    try: os.nice(nice) if nice else None
    except Exception: pass
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, wsize[0], wsize[1]); ppid = os.getppid(); gap = 1.0 / gui_fps if gui_fps > 0 else 0.0; t_last = 0.0; period = 1.0 / steady if steady > 0 else 0.0; nxt = 0; t_next = None; held = 0
        while not stop.is_set():
            if os.getppid() != ppid: break
            cur = seq.value; now = time.perf_counter()
            if period:
                if t_next is None:
                    if cur >= 3: nxt = cur - 2; t_next = now                      # start once 3 frames are buffered
                elif now >= t_next:
                    if cur - nxt > N - 3: nxt = cur - 2                           # fell too far behind: jump ahead (keeps the delay small)
                    if nxt <= cur: show(buf[nxt % N]); nxt += 1; shown_t.append(now)
                    else: held += 1                                               # nothing new yet: keep the last picture
                    t_next += period
                    if t_next < now - 0.25: t_next = now
            elif cur != last and now - t_last >= gap: last = cur; t_last = now; show(buf[cur % N]); shown_t.append(now)
            if cv2.waitKey(2) & 0xFF in (ord("q"), 27): stop.set()
    except Exception as e: print(f"[GUI] window process stopped: {e}", flush=True)
    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass
        if len(shown_t) > 10:
            d = np.diff(np.array(shown_t)) * 1e3; print(f"\n[GUI] window showed {len(shown_t)} pictures at {1000.0 / d.mean():.1f}/s (interval ms: median {np.median(d):.0f}, 95% {np.percentile(d, 95):.0f}, max {d.max():.0f}); repeated-picture ticks: {held}", flush=True)


# ------------------------------------------------------------------ model
def load_model(pte, threads):
    import torch
    from executorch.extension.pybindings import portable_lib as pl
    pl._unsafe_reset_threadpool(threads); return pl._load_for_executorch(pte), torch


def worker_main(pte, threads, S, conf, iou, agnostic, slots, tasks, results, stop, zones=None, zpoint="center", cluster=True, drop_person=True):
    """Own process: loads its own copy of the model, takes (frame id, slot, geometry) tasks, returns detections."""
    mod, torch = load_model(pte, threads); views = [np.frombuffer(s, np.float32).reshape(1, 3, S, S) for s in slots]; tens = [torch.from_numpy(v) for v in views]
    for _ in range(3): mod.forward([tens[0]])
    results.put(("ready", os.getpid()))
    while not stop.is_set():
        try: fid, slot, ow, oh, geo = tasks.get(timeout=0.2)
        except Exception: continue
        t0 = time.perf_counter(); outs = [o.numpy() for o in mod.forward([tens[slot]])]; t1 = time.perf_counter()
        boxes, scores, cids = decode_heads(outs, ow, oh, S, conf, iou, geo, agnostic, zones, zpoint, cluster, drop_person); t2 = time.perf_counter()
        results.put((fid, slot, boxes, scores, cids, (t1 - t0) * 1e3, (t2 - t1) * 1e3))


class ResizingFile:
    """A video FILE behaves like the live camera: pictures larger than the processing size are scaled down, and the full-resolution frame is kept for the zoom check."""
    def __init__(self, cap, max_w=704, max_h=576): self.cap, self.mw, self.mh = cap, max_w, max_h; self.last_full = None
    def read(self):
        ok, f = self.cap.read(); self.last_full = f if ok else None
        if ok and f is not None and (f.shape[1] > self.mw or f.shape[0] > self.mh): f = cv2.resize(f, (self.mw, self.mh), interpolation=cv2.INTER_LINEAR)
        return ok, f
    def isOpened(self): return self.cap.isOpened()
    def set(self, *a): return self.cap.set(*a)
    def get(self, *a): return self.cap.get(*a)
    def release(self): self.cap.release()


class Refiner:
    """Zoom check: re-classifies a vehicle from a tight crop of the FULL-RESOLUTION camera frame with a second (more accurate, larger-input) model. Called once per counted vehicle."""
    def __init__(self, pte, threads, S=256):
        self.mod, self.torch = load_model(pte, threads); self.S = S; self.pre = PreProc(S, True); self.buf = np.zeros((1, 3, S, S), np.float32); self.tens = self.torch.from_numpy(self.buf); self.ctx = {}
        for _ in range(2): self.mod.forward([self.tens])

    def _views(self, crop):
        """The vehicle seen four ways: as cut, with a little and with more surrounding, and mirrored (all with replicated borders)."""
        pad = lambda im, f: cv2.copyMakeBorder(im, *(int(f * max(im.shape[:2])),) * 4, cv2.BORDER_REPLICATE)
        return [crop]                                                                                       # ONE view is enough (105 vs 106 right of 108 with four) and four views stalled the video for ~0.3 s

    def __call__(self, t):
        """Second look with the larger (256) model at the moment of counting: the same crop as the saved photo (box + 6 %) from the full frame, four views, class evidence (score^2) added up.
        Measured on 108 hand-checked photos of the new camera: right 104 times vs 99 for the live 128 model (where both models agree: 96 of 96 right)."""
        full, wh = self.ctx.get("full"), self.ctx.get("wh")
        if full is None or wh is None: return None
        Wr, Hr = wh; Hf, Wf = full.shape[:2]; sx, sy = Wf / float(Wr), Hf / float(Hr); bw, bh = t["w"] * sx, t["h"] * sy; cx, cy = t["cx"] * sx, t["cy"] * sy; m = 0.06
        x0, y0 = int(max(0, cx - bw / 2 - m * bw)), int(max(0, cy - bh / 2 - m * bh)); x1, y1 = int(min(Wf, cx + bw / 2 + m * bw)), int(min(Hf, cy + bh / 2 + m * bh))
        if x1 - x0 < 24 or y1 - y0 < 24: return None
        ev = {}; top = {}
        for im in self._views(full[y0:y1, x0:x1]):
            ch, cw = im.shape[:2]; geo = self.pre(im, self.buf); outs = [o.numpy() for o in self.mod.forward([self.tens])]
            b, sc, ci = decode_heads(outs, cw, ch, self.S, 0.10, 0.45, geo, False, None, "center", True, True)
            for s_, c in zip(sc, ci): ev[c] = ev.get(c, 0.0) + s_ * s_; top[c] = max(top.get(c, 0.0), s_)
        if not ev: return None
        c = max(ev, key=ev.get); return (c, top[c])


# ------------------------------------------------------------------ camera: newest-frame only, no blocking lock, reconnect
class FreshRTSPCapture:
    """A thread keeps pulling the stream (so it never backs up) and always holds the NEWEST frame; read() returns it without copying. The thread never holds a lock while it waits for the camera,
    so the main loop is never blocked by it. Reconnects if the stream stalls."""
    def __init__(self, src, max_w=704, max_h=576, stall_s=6.0):
        self.src, self.max_w, self.max_h, self.stall_s = src, max_w, max_h, stall_s; self.cond = threading.Condition(); self.st = {"arrived": 0, "lost_cam": 0, "skipped": 0, "t0": time.time(), "t_prev": 0.0, "gap_ema": 0.0, "worst_gap": 0.0, "reconnects": 0}; self.frame = None; self.seq = 0; self.taken = 0; self.stopped = False; self.last_ok = time.time(); self.cap = None; self.native = None; self.full = None; self.last_full = None
        self._open(); self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _open(self):
        if self.cap is not None: self.cap.release()
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG); self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); self.cam_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0; print(f"  -> Connecting to video stream: {mask_url(self.src)} ...", flush=True)

    def _run(self):
        while not self.stopped:
            if not self.cap.grab():
                if time.time() - self.last_ok > self.stall_s: print("  ! stream stalled - reconnecting", flush=True); self.st["reconnects"] += 1; self._open(); self.last_ok = time.time()
                time.sleep(0.01); continue
            ok, f = self.cap.retrieve()
            if not ok or f is None: continue
            if self.native is None:
                self.native = (f.shape[1], f.shape[0]); big = f.shape[1] > self.max_w or f.shape[0] > self.max_h
                print(f"  -> camera picture size {f.shape[1]}x{f.shape[0]} = {'MAIN stream' if f.shape[1] >= 1280 else 'sub stream'}" + (f" (scaled to {self.max_w}x{self.max_h} for detection and display)" if big else ""), flush=True)
            orig = f
            if f.shape[1] > self.max_w or f.shape[0] > self.max_h: f = cv2.resize(f, (self.max_w, self.max_h), interpolation=cv2.INTER_LINEAR)
            self.last_ok = tn = time.time(); st = self.st; st["arrived"] += 1
            if st["t_prev"]:
                g = tn - st["t_prev"]; st["gap_ema"] = g if not st["gap_ema"] else 0.98 * st["gap_ema"] + 0.02 * g; st["worst_gap"] = max(st["worst_gap"], g)
                if g > 1.6 * max(st["gap_ema"], 0.02) and g > 0.07: st["lost_cam"] += max(0, int(round(g / max(st["gap_ema"], 0.02))) - 1)   # a hole in the arrivals = frames the camera / network did not deliver
            st["t_prev"] = tn
            with self.cond: self.frame = f; self.full = orig; self.seq += 1; self.cond.notify_all()

    def read(self, timeout=2.0):
        end = time.time() + timeout
        with self.cond:
            while self.seq == self.taken and time.time() < end: self.cond.wait(timeout=0.05)
            if self.seq == self.taken or self.frame is None: return False, None
            self.st["skipped"] += max(0, self.seq - self.taken - 1); self.taken = self.seq; self.last_full = self.full; return True, self.frame

    def isOpened(self): return self.cap is not None and self.cap.isOpened() and not self.stopped
    def release(self): self.stopped = True; self.t.join(timeout=1.0); self.cap is not None and self.cap.release()


# ------------------------------------------------------------------ drawing / reporting
def _match_score(a, b):
    """Overlap score of two xywh boxes: IoU, or - if one box sits mostly inside the other (an object entering/leaving the picture grows or shrinks a lot between detections) - the share of the smaller box that is covered."""
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]); ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    inter = iw * ih; aa, ab = a[2] * a[3], b[2] * b[3]; iou = inter / (aa + ab - inter + 1e-9); inside = inter / (min(aa, ab) + 1e-9)
    return max(iou, 0.9 * inside) if (iou > 0.3 or inside > 0.7) else 0.0


def smooth_boxes(dfids, dets, ftime, d, bs):
    """Boxes for frame d from the real detections around it: exactly the detection if d was detected; otherwise each object found in both the previous (A) and next (B) detection
    (best overlap first, same class) is moved/resized linearly by time between them; an object only in A is kept for the first half of the gap, one only in B for the second half."""
    i = bs.bisect_right(dfids, d); A = dfids[i - 1] if i > 0 else None; B = dfids[i] if i < len(dfids) else None
    if A is None: b, s, c, m = dets[B]; return list(b), list(s), list(c), m
    if A == d or B is None: b, s, c, m = dets[A]; return list(b), list(s), list(c), m
    bA, sA, cA, mA = dets[A]; bB, sB, cB, mB = dets[B]; w = (ftime[d] - ftime[A]) / max(ftime[B] - ftime[A], 1e-6); ob, os_, oc = [], [], []
    pairs = sorted(((_match_score(ba, bb), ia, jb) for ia, ba in enumerate(bA) for jb, bb in enumerate(bB) if cA[ia] == cB[jb]), reverse=True); ua, ub, link = set(), set(), []
    for q, ia, jb in pairs:
        if q > 0 and ia not in ua and jb not in ub: ua.add(ia); ub.add(jb); link.append((ia, jb))
    for ia, jb in link: ob.append(tuple(int(round((1 - w) * x + w * y)) for x, y in zip(bA[ia], bB[jb]))); os_.append((1 - w) * sA[ia] + w * sB[jb]); oc.append(cA[ia])
    if w < 0.5: ob += [tuple(int(x) for x in bA[ia]) for ia in range(len(bA)) if ia not in ua]; os_ += [sA[ia] for ia in range(len(bA)) if ia not in ua]; oc += [cA[ia] for ia in range(len(bA)) if ia not in ua]
    else: ob += [tuple(int(x) for x in bB[jb]) for jb in range(len(bB)) if jb not in ub]; os_ += [sB[jb] for jb in range(len(bB)) if jb not in ub]; oc += [cB[jb] for jb in range(len(bB)) if jb not in ub]
    return ob, os_, oc, (mA + mB) / 2


def annotate(frame, boxes, scores, cids, names, infer_ms, fps, tag, det=None, ids=None, hud3=None, hud=True):
    """Draws IN PLACE (the frame is not needed afterwards); the HUD banner is darkened only inside its own 48-px strip."""
    counts = Counter()
    for k_, ((x, y, w, h), s, c) in enumerate(zip(boxes, scores, cids)):
        n = names[c] if 0 <= c < len(names) else f"class_{c}"; counts[n] += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2); cv2.putText(frame, (f"#{ids[k_]} " if ids else "") + f"{n} {s:.2f}", (x, max(y - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    if not hud: return frame, counts                                                                          # clean picture: only boxes, ids and the counting line
    strip = frame[0:(66 if hud3 else 48)]; cv2.convertScaleAbs(strip, dst=strip, alpha=0.15)                  # same result as blending a black banner at 0.85
    cv2.putText(frame, f"INT8 Edge AI [{tag}] | Video {fps:.1f} FPS" + (f" | Detect {det:.1f}/s" if det is not None else "") + f" | Infer {infer_ms:.0f} ms | Objects {len(boxes)}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Counts: " + (", ".join(f"{k}: {v}" for k, v in counts.items()) if counts else "None"), (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if hud3: cv2.putText(frame, hud3, (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return frame, counts


def load_names(model_path):
    stem = os.path.splitext(os.path.basename(model_path))[0]
    for p in (f"{stem}.names.txt", os.path.join(os.path.dirname(os.path.abspath(model_path)), f"{stem}.names.txt")):
        if os.path.exists(p):
            lines = [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
            if lines: return lines
    return TRAFFIC_CLASSES


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Optimized ExecuTorch INT8 detection + counting (Raspberry Pi 4)")
    ap.add_argument("--source", default=None, help="video file, image/folder, or an rtsp:// URL (default: RTSP_URL env var / .rtsp.env)"); ap.add_argument("--weights", default=MODEL_PATH); ap.add_argument("--mode", default=RUN_MODE, choices=["SYNC_DETECTION", "ASYNC_STREAM_25FPS", "SMOOTH_25FPS"])
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE); ap.add_argument("--conf", type=float, default=CONF_THRESHOLD); ap.add_argument("--iou", type=float, default=IOU_THRESHOLD); ap.add_argument("--fps", type=float, default=TARGET_FPS)
    ap.add_argument("--workers", type=int, default=1, help="parallel detector processes (1 = original single-process path)"); ap.add_argument("--threads", type=int, default=0, help="XNNPACK threads per worker (default: 4 for one worker, else 4 // workers)")
    ap.add_argument("--agnostic-nms", action="store_true", help="old New.py behaviour (class-agnostic NMS)"); ap.add_argument("--letterbox", action="store_true", help="keep the aspect ratio (New.py stretches the frame to a square)")
    g = ap.add_mutually_exclusive_group(); g.add_argument("--gui", dest="gui", action="store_true", default=SHOW_GUI); g.add_argument("--no-gui", dest="gui", action="store_false")
    ap.add_argument("--loop", action="store_true", help="file source: restart at the end (demos on the Pi screen)")
    ap.add_argument("--web-overlay", action="store_true", help="draw the boxes, ids, counting line and zones on the picture of the WEBSITE too (default: the website shows the clean camera picture only)"); ap.add_argument("--track-log", action="store_true", help="write tracks.csv: one row per vehicle track (frames, detections, class, crossed?, counted?, reason) to find out why a vehicle was not counted"); ap.add_argument("--web", action="store_true", help="serve a simple web page (live detection video, report, video saving) instead of the local window"); ap.add_argument("--web-port", type=int, default=8000); ap.add_argument("--web-data", default=None, help="folder for the report database and the vehicle photos (default: web_data/ next to this script)"); ap.add_argument("--web-host", default="0.0.0.0", help="0.0.0.0 = reachable from the network, 127.0.0.1 = only this computer"); ap.add_argument("--rec-dir", default=None, help="folder for videos saved from the web page (default: recordings/ next to this script)"); ap.add_argument("--stream-fps", type=int, default=20, help="pictures per second sent to the web page"); ap.add_argument("--prescale", action="store_true", help="EXPERIMENTAL: resize each picture to the window size in the window process (can squash the picture on some desktops; off by default)"); ap.add_argument("--steady-fps", type=float, default=25.0, help="the window shows frames at this FIXED rate from a ~3-frame buffer (about 0.15 s behind live) so slow frames do not make it stutter; 0 = show each frame as soon as it is ready"); ap.add_argument("--gui-fps", type=float, default=0.0, help="redraw the window at most this many times a second (0 = every frame). Detection and counting keep their full speed; a lower value saves the CPU that a large window and the VNC server use"); ap.add_argument("--pin-cores", action="store_true", help="run the model on cores 0..N-2 and the window on the last core (use with --threads N-1), so the window size cannot slow detection"); ap.add_argument("--gui-nice", type=int, default=15, help="CPU priority penalty (0-19) of the window process: higher = detection keeps its speed when the window is large or slow (0 = same priority)"); ap.add_argument("--gui-inproc", action="store_true", help="draw the window inside the detection loop (old behaviour; a large window then slows detection)"); ap.add_argument("--window-size", default=None, help="initial window size WxH, e.g. 960x784 (you can still resize it with the mouse)"); ap.add_argument("--save-video", default=None, help="write the annotated output to this .mp4 file (to cross-check the detections and counts afterwards)"); ap.add_argument("--draw-line", action="store_true", help="open one camera frame, click the 2 end points of the COUNTING LINE, press s to save, then exit"); ap.add_argument("--count-line", default=None, help="JSON file with the counting line (default: count_line.json next to this script, if present)"); ap.add_argument("--count-point", choices=["center", "bottom"], default="center", help="which point of a box is tested against the counting line"); ap.add_argument("--flip-direction", action="store_true", help="swap which way is IN and OUT"); ap.add_argument("--refine-model", default=None, help="second model (e.g. the 256 model) used for a zoom re-classification of each vehicle from a crop of the full-resolution frame at the moment it is counted"); ap.add_argument("--hud", choices=["auto", "on", "off"], default="auto", help="text banner on the picture (FPS, counts, totals). auto = off with --web (the page shows that information beside the video; the saved recordings stay clean too), on for the local window"); ap.add_argument("--no-cluster", action="store_true", help="old behaviour: class-aware NMS (one box per class on the same object)"); ap.add_argument("--keep-person", action="store_true", help="keep person detections (by default pedestrians and riders are dropped: only vehicles are counted)"); ap.add_argument("--line-margin", type=float, default=0.025, help="dead-band on both sides of the counting line, as a fraction of the picture height: a vehicle counts only after it was clearly on one side and is then clearly on the other"); ap.add_argument("--park-memory-s", type=float, default=1800.0, help="a vehicle that stands still and is lost is remembered this many seconds (or until a moving vehicle passes that spot), so it is not counted twice if it is seen again on the same spot"); ap.add_argument("--min-hits", type=int, default=3, help="detections a vehicle needs before it may be counted"); ap.add_argument("--max-age", type=int, default=8, help="detections a vehicle may be missed before its track is dropped"); ap.add_argument("--draw-zones", action="store_true", help="open one camera frame, click the areas to IGNORE, press s to save, then exit"); ap.add_argument("--exclude-zones", default=None, help="JSON file with ignore-zones (default: exclude_zones.json next to this script, if present)"); ap.add_argument("--zone-point", choices=["center", "bottom"], default="center", help="which point of a box must be inside a zone for it to be ignored"); ap.add_argument("--hide-zones", action="store_true", help="do not draw the zone outlines"); ap.add_argument("--pipeline", action="store_true", help="SYNC mode: run the model in its own process so it overlaps with drawing/window/camera (boxes stay locked to their own frame; +1-2 frames delay)"); ap.add_argument("--profile", action="store_true", help="print where the display loop spends its time (ASYNC mode)"); ap.add_argument("--out", default=OUTPUT_DIR); ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--print-every", type=int, default=5)
    a = ap.parse_args(); S = a.imgsz; a.threads = a.threads or (4 if a.workers == 1 else max(1, 4 // a.workers))
    if a.web: a.gui = False                                                  # the web page replaces the local window
    if a.pin_cores and (os.cpu_count() or 1) >= 3:
        try: os.sched_setaffinity(0, set(range((os.cpu_count() or 4) - 1))); print(f" pinned: model on cores 0-{(os.cpu_count() or 4) - 2}, window on core {(os.cpu_count() or 4) - 1}")
        except Exception as e: print(f"[pin] not possible: {e}")
    src = a.source if a.source else load_rtsp_url()
    if not src: raise SystemExit("No source: pass --source, or set the RTSP_URL environment variable (or .rtsp.env next to this script).")
    if not os.path.exists(a.weights): raise FileNotFoundError(f"Model file '{a.weights}' not found.")
    zfile = a.exclude_zones or os.path.join(os.path.dirname(os.path.abspath(__file__)), "exclude_zones.json")
    if a.draw_zones:
        c0 = FreshRTSPCapture(src, DISPLAY_WIDTH, DISPLAY_HEIGHT) if str(src).startswith("rtsp://") else cv2.VideoCapture(src); ok0 = False
        for _ in range(60):
            ok0, f0 = c0.read()
            if ok0 and f0 is not None: break
        if not ok0: raise SystemExit("could not get a frame to draw on")
        draw_zones_tool(f0.copy(), zfile); c0.release(); return
    zones = load_zones(zfile) if os.path.exists(zfile) else []; lfile = a.count_line or os.path.join(os.path.dirname(os.path.abspath(__file__)), "count_line.json")
    if a.draw_line:
        c0 = FreshRTSPCapture(src, DISPLAY_WIDTH, DISPLAY_HEIGHT) if str(src).startswith("rtsp://") else cv2.VideoCapture(src); ok0 = False
        for _ in range(60):
            ok0, f0 = c0.read()
            if ok0 and f0 is not None: break
        if not ok0: raise SystemExit("could not get a frame to draw on")
        draw_line_tool(f0.copy(), lfile, zones, a.flip_direction); c0.release(); return
    cline = load_line(lfile) if os.path.exists(lfile) else None; trk = None; flip_dir = bool(a.flip_direction or (os.path.exists(lfile) and load_flip(lfile)))
    os.makedirs(a.out, exist_ok=True); samples = os.path.join(a.out, "sample_outputs"); os.makedirs(samples, exist_ok=True); names = load_names(a.weights); model_name = os.path.splitext(os.path.basename(a.weights))[0]
    show_hud = (a.hud == "on") or (a.hud == "auto" and not a.web)
    refiner = None
    if a.refine_model and os.path.exists(a.refine_model): refiner = Refiner(a.refine_model, a.threads); print(f" zoom re-classification at the line: ON ({os.path.basename(a.refine_model)})", flush=True)
    groups_map = {i: (0 if n.lower() in ("two_wheeler", "cycle", "person") else 1 if n.lower() == "auto_rickshaw" else 2) for i, n in enumerate(names)}
    def make_tracker(line_):
        t_ = VehicleTracker(line_, flip_dir, a.count_point, a.min_hits, a.max_age, a.line_margin, a.park_memory_s, groups_map); t_.refiner = refiner; return t_
    trk = make_tracker(cline) if cline else None
    is_rtsp = str(src).startswith("rtsp://"); is_dir = os.path.isdir(src); is_img = os.path.isfile(src) and src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    print("=" * 70); print(f" Optimized ExecuTorch INT8 detection | mode {a.mode} | workers {a.workers} x {a.threads} thread(s) | NMS {'agnostic (old)' if a.agnostic_nms else 'class-aware'} | {'letterbox' if a.letterbox else 'stretch'}"); print(f" Source: {mask_url(src)} | Model: {a.weights} | {S}x{S} | GUI {'on' if a.gui else 'off'} | ignore-zones: {len(zones)}" + (f" ({zfile})" if zones else "") + (f" | counting line: ON ({lfile})" if trk else " | counting line: off (draw one with --draw-line)")); print("=" * 70)
    csvf = open(os.path.join(a.out, "detection_counts.csv"), "w", newline="", encoding="utf-8"); cw = csv.writer(csvf); cw.writerow(["Frame_or_Image", "Mode", "FPS", "Infer_Latency_ms", "Total_Count", "Category_Counts"])
    from collections import defaultdict; PT = defaultdict(float); nres = [0]; t_end = [0.0]; pre = PreProc(S, a.letterbox); totals = Counter(); lat = []; saved = 0; shown = 0; fps_hist = deque(maxlen=40); det_t = deque(maxlen=200); t_wall0 = None
    win = f"Edge AI ({a.mode})"
    if a.gui and a.gui_inproc:
        try: cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, *(tuple(int(x) for x in a.window_size.lower().split("x")) if a.window_size else (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
        except Exception as e: print(f"[GUI] cannot open a window ({e}); continuing without it"); a.gui = False

    fdim = [DISPLAY_WIDTH, DISPLAY_HEIGHT]; xcsv = open(os.path.join(a.out, "crossings.csv"), "w", newline="", encoding="utf-8") if trk else None; xw = csv.writer(xcsv) if trk else None
    if trk: xw.writerow(["Time", "Frame", "Track_ID", "Class", "Direction"])
    tcsv = open(os.path.join(a.out, "tracks.csv"), "w", newline="", encoding="utf-8") if (trk and a.track_log) else None; tw = csv.writer(tcsv) if tcsv else None
    if tw: tw.writerow(["Track_ID", "First_frame", "Last_frame", "Detections", "Class", "Avg_conf", "Crossed", "Counted", "Reason"])
    def log_tracks():
        if tw:
            for r_ in trk.done: tw.writerow([r_[0], r_[1], r_[2], r_[3], names[r_[4]] if 0 <= r_[4] < len(names) else r_[4], *r_[5:]]); tcsv.flush()
        trk.done.clear()
    def count_step(b, s_, c_, fid_):
        """Feed one frame's REAL detections to the tracker; print a line in the terminal for every vehicle that crosses the counting line."""
        if not trk: return None
        ids_, ev = trk.update(b, s_, c_, fdim[0], fdim[1], fid_)
        for (tid, cls, d_, f_) in ev:
            n_ = names[cls] if 0 <= cls < len(names) else f"class_{cls}"; i_, o_ = trk.totals(); allc = Counter(); allc.update(trk.counts["IN"]); allc.update(trk.counts["OUT"]); tm = time.strftime("%H:%M:%S")
            xw.writerow([tm, f_, tid, n_, d_]); xcsv.flush(); web_evt(tm, f_, tid, n_, d_)
            print(f"\n  [{tm}] COUNT  {n_:<16} #{tid:<4} {d_:<3} | total {i_ + o_:<4} (IN {i_} / OUT {o_}) | " + ", ".join(f"{(names[k] if 0 <= k < len(names) else k)} {v}" for k, v in allc.most_common()), flush=True)
        log_tracks(); return ids_

    web = {"proc": None, "t_status": 0.0}
    evq = []
    def web_evt(tm, f_, tid, n_, d_):
        if web["proc"] is not None: evq.append((tm, f_, tid, n_, d_, trk.info.get(tid) if trk else None))
    def flush_events(frame, tag_):
        """Send each newly counted vehicle to the web pages, with a small photo cut from the SAME frame (only in SYNC mode, where the frame is the one that was detected)."""
        while evq:
            tm, f_, tid, n_, d_, info = evq.pop(0); jpg = None; conf = info[0] if info else None
            if info and tag_ == "SYNC" and frame is not None:
                src_ = frame
                try: lf_ = cap.last_full                                                                  # the FULL-resolution frame of the same picture (1080p on the main stream): far sharper photos
                except Exception: lf_ = None
                if lf_ is not None and lf_.shape[0] > frame.shape[0]: src_ = lf_
                sx_, sy_ = src_.shape[1] / float(frame.shape[1]), src_.shape[0] / float(frame.shape[0]); x, y, w, h = info[1]; H_, W_ = src_.shape[:2]; mx_, my_ = 0.06 * w, 0.06 * h
                x0, y0, x1, y1 = int(max(0, (x - mx_) * sx_)), int(max(0, (y - my_) * sy_)), int(min(W_, (x + w + mx_) * sx_)), int(min(H_, (y + h + my_) * sy_))
                if x1 - x0 > 8 and y1 - y0 > 8:
                    crop = src_[y0:y1, x0:x1]; k_ = min(1.0, 360.0 / max(crop.shape[:2]))
                    if k_ < 1.0: crop = cv2.resize(crop, (max(1, int(crop.shape[1] * k_)), max(1, int(crop.shape[0] * k_))), interpolation=cv2.INTER_AREA)
                    ok_, bb = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90]); jpg = bb.tobytes() if ok_ else None
            if web["proc"] is not None:
                try: web["q"].put_nowait(("event", {"time": tm, "frame": int(f_), "id": int(tid), "class": n_, "dir": d_, "conf": conf, "jpg": jpg}))
                except Exception: pass
    def stream_label():
        try: n_ = cap.native
        except Exception: n_ = None
        return None if not n_ else ("%s stream %dx%d" % ("Main" if n_[0] >= 1280 else "Sub", n_[0], n_[1]))
    def web_status(fps_, det_, infer_, nobj, tag_):
        ci = {}; co = {}; ti = to = 0
        if trk:
            ti, to = trk.totals(); ci = {(names[k] if 0 <= k < len(names) else f"class_{k}"): v for k, v in trk.counts["IN"].items()}; co = {(names[k] if 0 <= k < len(names) else f"class_{k}"): v for k, v in trk.counts["OUT"].items()}
        try: web["q"].put_nowait(("status", {"fps": round(fps_, 1), "detect": round(det_, 1), "infer_ms": round(infer_, 1), "objects": int(nobj), "mode": {"SYNC": "every frame detected", "SMOOTH": "smooth video"}.get(tag_, tag_), "source": "live camera" if is_rtsp else os.path.basename(str(src)), "stream": stream_label(),
                                                 "line": bool(trk), "roi_live": roi_live, "counts_in": ci, "counts_out": co, "total_in": ti, "total_out": to}))
        except Exception: pass
    if a.web:
        from web_ui import web_main
        wctx = mp.get_context("spawn"); web["shm"] = wctx.RawArray("B", GUI_RING * DISPLAY_HEIGHT * DISPLAY_WIDTH * 3); web["seq"] = wctx.RawValue("i", 0); web["stop"] = wctx.Event(); web["q"] = wctx.Queue(maxsize=500); web["cmd"] = wctx.Queue(); web["cshm"] = wctx.RawArray("B", DISPLAY_HEIGHT * DISPLAY_WIDTH * 3); web["cseq"] = wctx.RawValue("i", 0); web["t_clean"] = 0.0
        web["views"] = np.frombuffer(web["shm"], np.uint8).reshape(GUI_RING, DISPLAY_HEIGHT, DISPLAY_WIDTH, 3); web["cview"] = np.frombuffer(web["cshm"], np.uint8).reshape(DISPLAY_HEIGHT, DISPLAY_WIDTH, 3); rec_dir = a.rec_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        web["proc"] = wctx.Process(target=web_main, args=(web["shm"], (GUI_RING, DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), web["seq"], web["stop"], web["q"], web["cmd"], {"host": a.web_host, "port": a.web_port, "rec_dir": rec_dir, "data_dir": a.web_data or os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_data"), "stream_fps": a.stream_fps, "jpeg_quality": 75, "classes": [n_ for n_ in names if n_ != "person"], "line_file": lfile, "zones_file": zfile, "ref_size": [DISPLAY_WIDTH, DISPLAY_HEIGHT]}, web["cshm"], web["cseq"]), daemon=True); web["proc"].start()
        try: sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sk.connect(("10.255.255.255", 1)); ip = sk.getsockname()[0]; sk.close()
        except Exception: ip = "<this-computer>"
        print(f" Web page: http://{ip}:{a.web_port}   (videos saved from the page go to {rec_dir})", flush=True)
    def reload_roi():
        """The web page saved a new counting line / ignore-zones: load them into the RUNNING detector (single-process SYNC mode)."""
        nonlocal zones, cline, trk, flip_dir
        try:
            zones = load_zones(zfile) if os.path.exists(zfile) else []; new_line = load_line(lfile) if os.path.exists(lfile) else None; flip_dir = bool(a.flip_direction or (os.path.exists(lfile) and load_flip(lfile))); cline = new_line
            if new_line is None: trk = None
            elif trk is None: trk = make_tracker(new_line)
            else:
                trk.line = new_line; trk.flip = flip_dir; trk.mem = []                                                                    # vehicles are judged against the NEW line from now on
                for t_ in trk.tracks: t_["side"] = None; t_["side_pt"] = t_["pt"]; t_["side_d"] = 0.0; t_["cross"] = t_["cross"] if t_["counted"] else None
            print(f"\n  [ROI] counting line and ignore-zones reloaded: line {'set' if cline else 'none'}, {len(zones)} zone(s), IN direction {'swapped' if flip_dir else 'normal'}", flush=True)
        except Exception as e: print(f"\n  [ROI] reload failed ({e}); the running settings are unchanged", flush=True)
    vw = [None]; gp = {"proc": None, "dead": False}; roi_live = (a.workers == 1 and a.mode == "SYNC_DETECTION" and not a.pipeline)
    def present(frame, boxes, scores, cids, infer_ms, tag, fid, ids=None):
        nonlocal saved, shown, t_wall0
        if web["proc"] is not None and time.perf_counter() - web["t_clean"] >= 1.0:                          # a CLEAN picture (no boxes, no line) for the line / zone editor on the web page
            web["t_clean"] = time.perf_counter(); im_c = frame if frame.shape[:2] == (DISPLAY_HEIGHT, DISPLAY_WIDTH) else cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)); web["cview"][:] = im_c; web["cseq"].value += 1
        if evq: flush_events(frame, tag)
        now = time.perf_counter(); t_wall0 = t_wall0 or now; t_end[0] = now; fps_hist.append(now); fps = (len(fps_hist) - 1) / (fps_hist[-1] - fps_hist[0]) if len(fps_hist) > 1 else 0.0
        if is_rtsp and hasattr(cap, "st") and now - web.setdefault("t_cam", now) >= 30.0:                    # every 30 s: how many frames the camera delivered / lost and how many the detector skipped
            web["t_cam"] = now; st_ = cap.st; el_ = max(time.time() - st_["t0"], 1e-6); st_["t0"] = time.time(); arr_, lost_, skp_ = st_["arrived"], st_["lost_cam"], st_["skipped"]; st_["arrived"] = st_["lost_cam"] = st_["skipped"] = 0
            try: t_c = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
            except Exception: t_c = -1.0
            cf_ = getattr(cap, "cam_fps", 0.0); lost_ = max(0, int(round(cf_ * el_ - arr_))) if 5.0 < cf_ < 120.0 else lost_                              # loss = frames the camera says it sends minus frames that arrived (uneven arrival is not loss)
            print(f"\n  [CAM] last {el_:.0f}s: camera sends {cf_:.0f} fps, delivered {arr_ / el_:.1f} fps, lost on the way {lost_} ({100.0 * lost_ / max(arr_ + lost_, 1):.1f}%), skipped by the detector {skp_} ({100.0 * skp_ / max(arr_, 1):.1f}%), worst gap {st_['worst_gap'] * 1000:.0f} ms, reconnects {st_['reconnects']}, CPU {t_c:.1f} C", flush=True); st_["worst_gap"] = 0.0
        clean_web = frame.copy() if (web["proc"] is not None and not a.web_overlay) else None                                  # the website gets the picture as the camera sends it (no boxes, no line, no zones)
        if trk: draw_count_line(frame, cline, flip_dir)
        if zones and not a.hide_zones: cv2.polylines(frame, zones_px(zones, frame.shape[1], frame.shape[0]), True, (0, 0, 255), 1)
        det = fps if tag == "SYNC" else float(sum(1 for t_ in det_t if now - t_ <= 1.0)); hud3 = None
        if trk:
            i_, o_ = trk.totals(); both = Counter(); both.update(trk.counts["IN"]); both.update(trk.counts["OUT"])
            hud3 = (f"LINE  IN {i_}  OUT {o_}  total {i_ + o_}  | " + ", ".join(f"{(names[k] if 0 <= k < len(names) else k)} {v}" for k, v in both.most_common()))[:110]
        ann, counts = annotate(frame, boxes, scores, cids, names, infer_ms, fps, tag, det, ids, hud3, show_hud); totals.update(counts); lat.append(infer_ms); shown += 1; cs = ", ".join(f"{k}:{v}" for k, v in counts.items()) or "None"
        cw.writerow([f"frame_{fid:04d}", tag, f"{fps:.1f}", f"{infer_ms:.2f}", len(boxes), cs])
        if shown % a.print_every == 0 or not is_rtsp and shown <= 3: print(f"\r  Frame {fid:04d} | {fps:4.1f} FPS | infer {infer_ms:5.1f} ms | objects {len(boxes):2d} | {cs[:60]:<60}", end="" if is_rtsp else "\n", flush=True)
        if boxes and saved < MAX_SAMPLE_SAVES and (saved == 0 or fid % 25 == 0): cv2.imwrite(os.path.join(samples, f"sample_detection_output_{saved + 1}.jpg"), ann); saved += 1
        if web["proc"] is not None:
            src_w = clean_web if clean_web is not None else ann; im_w = src_w if src_w.shape[:2] == (DISPLAY_HEIGHT, DISPLAY_WIDTH) else cv2.resize(src_w, (DISPLAY_WIDTH, DISPLAY_HEIGHT)); iw_ = (web["seq"].value + 1) % GUI_RING; web["views"][iw_][:] = im_w; web["seq"].value += 1
            if now - web["t_status"] >= 0.3: web["t_status"] = now; web_status(fps, det, infer_ms, len(boxes), tag)
            try: cmd_ = web["cmd"].get_nowait()
            except Exception: cmd_ = None
            if cmd_ and cmd_[0] == "roi_reload" and roi_live: reload_roi()
            if cmd_ and cmd_[0] == "reset":
                evq.clear()
                if trk: trk.counts = {"IN": Counter(), "OUT": Counter()}
        if a.save_video:
            if vw[0] is None: vw[0] = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), max(5.0, min(30.0, fps or a.fps)), (ann.shape[1], ann.shape[0]))
            vw[0].write(ann)
        if a.gui and a.gui_inproc:
            cv2.imshow(win, ann)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27): return False
        elif a.gui:
            if gp["proc"] is None and not gp["dead"]:
                gp["ctx"] = mp.get_context("spawn"); gp["shm"] = gp["ctx"].RawArray("B", GUI_RING * DISPLAY_HEIGHT * DISPLAY_WIDTH * 3); gp["seq"] = gp["ctx"].RawValue("i", 0); gp["stop"] = gp["ctx"].Event()
                gp["views"] = np.frombuffer(gp["shm"], np.uint8).reshape(GUI_RING, DISPLAY_HEIGHT, DISPLAY_WIDTH, 3)
                try: ws = tuple(int(x) for x in a.window_size.lower().split("x")) if a.window_size else (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                except Exception: ws = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                gp["proc"] = gp["ctx"].Process(target=gui_main, args=(gp["shm"], (GUI_RING, DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), gp["seq"], gp["stop"], win, ws, a.gui_nice, (os.cpu_count() or 4) - 1 if a.pin_cores else None, a.gui_fps, a.steady_fps, a.prescale), daemon=True); gp["proc"].start()
            if gp["proc"] is not None:
                if gp["stop"].is_set(): return False
                if not gp["proc"].is_alive(): print("\n[GUI] window process ended - continuing without a window", flush=True); gp["dead"] = True; gp["proc"] = None
                else:
                    im = ann if ann.shape[:2] == (DISPLAY_HEIGHT, DISPLAY_WIDTH) else cv2.resize(ann, (DISPLAY_WIDTH, DISPLAY_HEIGHT)); i_ = (gp["seq"].value + 1) % GUI_RING; gp["views"][i_][:] = im; gp["seq"].value += 1
        return True

    if is_dir or is_img:                                                            # ---- images: always single process
        mod, torch = load_model(a.weights, a.threads); buf = np.zeros((1, 3, S, S), np.float32); tt = torch.from_numpy(buf)
        paths = [src] if is_img else sorted(sum((glob.glob(os.path.join(src, e)) for e in ("*.jpg", "*.jpeg", "*.png")), []))
        for k, p in enumerate(paths, 1):
            f = cv2.imread(p)
            if f is None: continue
            h, w = f.shape[:2]; t0 = time.perf_counter(); geo = pre(f, buf); outs = [o.numpy() for o in mod.forward([tt])]; b, s, c = decode_heads(outs, w, h, S, a.conf, a.iou, geo, a.agnostic_nms, zones, a.zone_point, not a.no_cluster, not a.keep_person); ms = (time.perf_counter() - t0) * 1e3
            cv2.imwrite(os.path.join(samples, f"annotated_{os.path.basename(p)}"), annotate(f, b, s, c, names, ms, 1000 / ms, "IMAGE")[0]); counts = Counter(names[i] for i in c); totals.update(counts); lat.append(ms)
            cw.writerow([os.path.basename(p), "IMAGE", f"{1000 / ms:.1f}", f"{ms:.2f}", len(b), ", ".join(f"{k}:{v}" for k, v in counts.items()) or "None"]); print(f"  [{k:02d}/{len(paths):02d}] {os.path.basename(p):<22} {ms:5.1f} ms | {len(b):2d} objects | {dict(counts)}")
        saved = len(paths); wall = sum(lat) / 1e3
    else:
        cap = FreshRTSPCapture(src, DISPLAY_WIDTH, DISPLAY_HEIGHT) if is_rtsp else ResizingFile(cv2.VideoCapture(int(src) if str(src).isdigit() else src), DISPLAY_WIDTH, DISPLAY_HEIGHT)
        if not cap.isOpened():
            if is_rtsp and a.web: print("  ! camera not reachable yet - the web pages stay up (CAMERA OFFLINE) and the program keeps retrying", flush=True)
            else: raise RuntimeError(f"Could not open the source: {mask_url(src)}")
        def rd():                                                               # --loop restarts a FILE at its end; camera streams are unaffected
            ok, f = cap.read()
            if (not ok or f is None) and a.loop and not is_rtsp: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f = cap.read()
            if ok and f is not None: fdim[0], fdim[1] = f.shape[1], f.shape[0]
            return ok, f
        ctx = mp.get_context("spawn") if (a.workers > 1 or a.pipeline or a.mode in ("ASYNC_STREAM_25FPS", "SMOOTH_25FPS")) else None
        try:
            if ctx is None:                                                         # ---- single process, locked boxes (the New.py SYNC path, optimized)
                mod, torch = load_model(a.weights, a.threads); buf = np.zeros((1, 3, S, S), np.float32); tt = torch.from_numpy(buf)
                for _ in range(3): mod.forward([tt])
                fid = 0
                while True:
                    ok, f = rd()
                    if not ok or f is None:
                        if is_rtsp: continue
                        break
                    fid += 1; h, w = f.shape[:2]; t0 = time.perf_counter(); geo = pre(f, buf); outs = [o.numpy() for o in mod.forward([tt])]; b, s, c = decode_heads(outs, w, h, S, a.conf, a.iou, geo, a.agnostic_nms, zones, a.zone_point, not a.no_cluster, not a.keep_person); ms = (time.perf_counter() - t0) * 1e3
                    if refiner is not None: lf_ = getattr(cap, "last_full", None); refiner.ctx["full"] = lf_ if lf_ is not None else f; refiner.ctx["wh"] = (w, h)
                    ids_ = count_step(b, s, c, fid)
                    if not present(f, b, s, c, ms, "SYNC", fid, ids_) or (a.max_frames and fid >= a.max_frames): break
            else:                                                                   # ---- worker processes
                nw = max(1, a.workers); NS = nw + 2; slots = [ctx.RawArray("f", 3 * S * S) for _ in range(NS)]; views = [np.frombuffer(s, np.float32).reshape(1, 3, S, S) for s in slots]
                tasks, results, stop = ctx.Queue(maxsize=NS), ctx.Queue(), ctx.Event()
                procs = [ctx.Process(target=worker_main, args=(a.weights, a.threads, S, a.conf, a.iou, a.agnostic_nms, slots, tasks, results, stop, zones, a.zone_point, not a.no_cluster, not a.keep_person), daemon=True) for _ in range(nw)]; [p.start() for p in procs]; ready = 0
                while ready < nw: m = results.get(); ready += (m[0] == "ready")
                free = deque(range(NS)); pend = {}; done = {}; fid = 0; nxt = 1; eof = False; latest = ([], [], [], 0.0); keep = True; last_t = time.perf_counter()
                import bisect as _bs; dets = {}; dfids = []; ftime = {}; fbuf = {}; lat_e = 0.15; int_e = 0.07; last_arr = None; last_shown = 0   # SMOOTH_25FPS state
                while keep:
                    if a.mode == "SYNC_DETECTION":
                        while free and not eof and len(pend) < NS - 1:
                            ok, f = rd()
                            if not ok or f is None:
                                if is_rtsp: break
                                eof = True; break
                            fid += 1; sl = free.popleft(); h, w = f.shape[:2]; geo = pre(f, views[sl]); pend[fid] = (f, sl); tasks.put((fid, sl, w, h, geo))
                    elif a.mode == "SMOOTH_25FPS":                                              # SMOOTH: show the video ~0.25 s behind live at camera rate; boxes are interpolated between the two real detections that bracket each frame
                        ok, f = rd(); tnow = time.perf_counter()
                        if not ok or f is None:
                            if is_rtsp: continue
                            eof = True
                        else:
                            fid += 1; ftime[fid] = tnow; fbuf[fid] = f
                            if free: sl = free.popleft(); h, w = f.shape[:2]; geo = pre(f, views[sl]); tasks.put((fid, sl, w, h, geo))
                            dly = min(0.7, max(0.12, lat_e + 1.5 * int_e + 0.02)); cand = [k for k in fbuf if ftime[k] <= tnow - dly]; dd = max(cand) if cand else 0
                            if dd > last_shown and dfids:
                                bx, sc, ci, ms_ = smooth_boxes(dfids, dets, ftime, dd, _bs); last_shown = dd
                                keep = present(fbuf[dd], bx, sc, ci, ms_, "SMOOTH", dd) and not (a.max_frames and shown >= a.max_frames)
                                for k in [k for k in fbuf if k < dd]: del fbuf[k]
                                j = _bs.bisect_right(dfids, dd) - 1
                                for k in dfids[:max(j, 0)]: dets.pop(k, None)
                                del dfids[:max(j, 0)]
                                for k in [k for k in ftime if k < min(dd, dfids[0] if dfids else dd)]: del ftime[k]
                            sleep = 1.0 / a.fps - (time.perf_counter() - last_t); sleep > 0 and time.sleep(sleep); last_t = time.perf_counter()
                    else:                                                           # ASYNC: show every camera frame at once with the newest available detections
                        _t0 = time.perf_counter(); ok, f = rd(); _t1 = time.perf_counter(); PT['read'] += _t1 - _t0
                        if not ok or f is None:
                            if is_rtsp: continue
                            eof = True
                        else:
                            fid += 1
                            if free: sl = free.popleft(); h, w = f.shape[:2]; geo = pre(f, views[sl]); pend[fid] = (None, sl); tasks.put((fid, sl, w, h, geo))
                            _t2 = time.perf_counter(); PT['preprocess+submit'] += _t2 - _t1
                            keep = present(f, *latest, "25FPS", fid) and not (a.max_frames and fid >= a.max_frames); _t3 = time.perf_counter(); PT['present(annotate+csv)'] += _t3 - _t2
                            sleep = 1.0 / a.fps - (time.perf_counter() - last_t); sleep > 0 and time.sleep(sleep); last_t = time.perf_counter(); PT['sleep'] += max(sleep, 0); PT['frames'] += 1
                    import queue as _q
                    got = False
                    try:
                        r = results.get(timeout=0.02 if (a.mode == "SYNC_DETECTION" and (pend or eof)) else 0.0005)               # wait briefly for ONE result, then drain whatever else is ready without waiting
                        while True:
                            fid_r, sl, b, s, c, ex, dc = r; free.append(sl); done[fid_r] = (b, s, c, ex + dc); got = True; nres[0] += 1; det_t.append(time.perf_counter())
                            if a.mode != "SYNC_DETECTION" and a.workers == 1: count_step(b, s, c, fid_r)
                            if a.mode == "SMOOTH_25FPS":
                                tr = time.perf_counter(); dets[fid_r] = (b, s, c, ex + dc); _bs.insort(dfids, fid_r); done.pop(fid_r, None); lat_e = 0.9 * lat_e + 0.1 * (tr - ftime.get(fid_r, tr))
                                if last_arr is not None: int_e = 0.9 * int_e + 0.1 * (tr - last_arr)
                                last_arr = tr
                            if a.mode != "SYNC_DETECTION": latest = (b, s, c, ex + dc)              # ASYNC: the newest FINISHED result
                            r = results.get_nowait()
                    except _q.Empty: pass
                    if a.mode == "SYNC_DETECTION":
                        while nxt in done and nxt in pend:
                            f, sl = pend.pop(nxt); b, s, c, ms = done.pop(nxt)
                            ids_ = count_step(b, s, c, nxt)
                            if not present(f, b, s, c, ms, "SYNC", nxt, ids_) or (a.max_frames and nxt >= a.max_frames): keep = False; break
                            nxt += 1
                        if eof and nxt > fid: break
                        if a.max_frames and nxt > a.max_frames: break
                    elif eof: break
                stop.set(); [p.join(timeout=3) for p in procs]; [p.terminate() for p in procs if p.is_alive()]
        except KeyboardInterrupt: print("\nInterrupted by user.")
        finally:
            cap.release(); (a.gui and a.gui_inproc) and cv2.destroyAllWindows()
            if web["proc"] is not None:
                web["stop"].set(); web["proc"].join(timeout=5)
                if web["proc"].is_alive(): web["proc"].terminate()
            if gp["proc"] is not None:
                gp["stop"].set(); gp["proc"].join(timeout=2)
                if gp["proc"].is_alive(): gp["proc"].terminate()
        wall = (t_end[0] - t_wall0) if t_wall0 else 0.0                       # the clock stops at the last displayed frame (worker shutdown is not counted)
    if a.profile and PT['frames']: print(' display-loop profile (ms per displayed frame): ' + ', '.join(f'{k} {1000 * v / PT["frames"]:.1f}' for k, v in PT.items() if k != 'frames') + f' | frames {int(PT["frames"])}')
    if trk:
        for (tid, cls, d_, f_) in trk.flush(shown): n_ = names[cls] if 0 <= cls < len(names) else f"class_{cls}"; xw.writerow([time.strftime("%H:%M:%S"), f_, tid, n_, d_]); print(f"\n  [{time.strftime('%H:%M:%S')}] COUNT  {n_:<16} #{tid:<4} {d_:<3} | (counted when the track ended)", flush=True)
        log_tracks(); tcsv and tcsv.close()
    csvf.close(); xcsv and xcsv.close(); vw[0] and vw[0].release(); avg = float(np.mean(lat)) if lat else 0.0; thr = (shown - 1) / wall if (wall > 0 and shown > 1 and not (is_dir or is_img)) else (1000 / avg if avg else 0.0); overall = ", ".join(f"{k}: {v}" for k, v in totals.most_common()) or "None"
    rep = ("\n" + "=" * 70 + "\n DELIVERABLE & BENCHMARK REPORT (optimized)\n" + "=" * 70 + f"\n Target Hardware           : Raspberry Pi 4 (Cortex-A72)\n Runtime & Delegate        : ExecuTorch + XNNPACK (INT8)\n Mode                      : {a.mode}, {a.workers} worker(s) x {a.threads} thread(s)\n"
           f" Model                     : {model_name}\n Input Resolution          : {S}x{S} ({'letterbox' if a.letterbox else 'stretch'})\n NMS                       : {'class-agnostic' if a.agnostic_nms else 'class-aware'} (IoU {a.iou}, conf {a.conf})\n Processed Frames / Images : {len(lat)}\n"
           f" Per-frame latency         : {avg:.2f} ms (pre-process + model + decode, measured in the worker)\n Measured throughput       : {thr:.2f} frames/s (wall clock, whole pipeline)\n AI results completed      : {nres[0]} ({(nres[0] / wall if wall > 0 else 0):.2f} detections/s)\n Total Detected Counts     : {overall}\n Detection Log (CSV)       : {os.path.join(a.out, 'detection_counts.csv')}\n Saved Detection Outputs   : {saved} images in '{samples}/'\n" + "=" * 70 + "\n")
    if trk:
        i_, o_ = trk.totals(); rows = sorted(set(trk.counts["IN"]) | set(trk.counts["OUT"]), key=lambda k: -(trk.counts["IN"][k] + trk.counts["OUT"][k]))
        rep += " VEHICLES COUNTED AT THE LINE (each vehicle once): IN %d / OUT %d / total %d   (list: %s)\n" % (i_, o_, i_ + o_, os.path.join(a.out, "crossings.csv")) + ((" ZOOM RE-CLASSIFICATION: %d vehicles checked, %d answered, %d classes changed\n" % (trk.rstat["checked"], trk.rstat["answered"], trk.rstat["changed"])) if refiner is not None else "") + "".join("   %-18s IN %-4d OUT %-4d\n" % (names[k] if 0 <= k < len(names) else k, trk.counts["IN"][k], trk.counts["OUT"][k]) for k in rows) + "=" * 70 + "\n"
    print(rep); open(os.path.join(a.out, "benchmark_report.txt"), "w", encoding="utf-8").write(rep)


if __name__ == "__main__":
    main()
