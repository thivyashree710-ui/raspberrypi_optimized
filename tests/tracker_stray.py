import sys, importlib.util
spec = importlib.util.spec_from_file_location("N", sys.argv[1]); N = importlib.util.module_from_spec(spec); sys.modules["N"] = N; spec.loader.exec_module(N)
W, H = 704, 576; Y0 = 0.5 * H; G = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
def box(cx, cy, w, h): return (int(cx - w / 2), int(cy - h / 2), w, h)
def scenario(stray_off):
    trk = N.VehicleTracker([(0.0, 0.5), (1.0, 0.5)], False, "center", 3, 8, groups=G); ev = []; fid = 0
    def feed(d):
        nonlocal fid; fid += 1; return trk.update([x[0] for x in d], [x[1] for x in d], [x[2] for x in d], W, H, fid, fid / 20.0)[1]
    y = 150.0
    for k in range(120):                                                                          # a real auto drives across the line (61+ detections)
        d = [(box(300, y, 110, 150), 0.9, 2)]
        if k == 38: d.append((box(300 + stray_off, Y0 - 25, 70, 90), 0.4, 2))                         # a stray detection on the auto: one frame just BEFORE the line ...
        if k == 39: d.append((box(300 + stray_off, Y0 + 32, 70, 90), 0.4, 2))                         # ... and one just AFTER it (so the stray track 'crosses' with only 2 detections)
        ev += feed(d); y += 3.5
    for _ in range(20): ev += feed([(box(300, y, 110, 150), 0.9, 2)])
    return len(ev)
for off in (0, 10, 25):
    print("stray offset %2d px -> counted %d (expected 1)" % (off, scenario(off)))

def early_stray(direction, off):
    """stray track with 3 detections crosses the line (OUT or IN) BEFORE the real vehicle is counted - exactly the white-car case"""
    trk = N.VehicleTracker([(0.0, 0.5), (1.0, 0.5)], False, "center", 3, 8, groups=G); ev = []; fid = 0
    def feed(d):
        nonlocal fid; fid += 1; return trk.update([x[0] for x in d], [x[1] for x in d], [x[2] for x in d], W, H, fid, fid / 20.0)[1]
    y = 150.0; ys = (Y0 + 30, Y0, Y0 - 25) if direction == "OUT" else (Y0 - 25, Y0, Y0 + 30)
    for k in range(120):
        d = [(box(300, y, 110, 150), 0.9, 3)]
        if 36 <= k <= 38: d.append((box(300 + off, ys[k - 36], 70, 90), 0.4, 3))
        ev += feed(d); y += 3.5
    for _ in range(20): ev += feed([(box(300, y, 110, 150), 0.9, 3)])
    return len(ev)
for d in ("OUT", "IN"):
    for off in (0, 15): print("early stray %-3s offset %2d -> counted %d (expected 1)" % (d, off, early_stray(d, off)))
