import sys, importlib.util, random
spec = importlib.util.spec_from_file_location("N", sys.argv[1]); N = importlib.util.module_from_spec(spec); sys.modules["N"] = N; spec.loader.exec_module(N)
W, H = 704, 576; Y0 = 0.5 * H; G = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
def box(cx, cy, w, h): return (int(cx - w / 2), int(cy - h / 2), w, h)
def run(occ_start, occ_len, speed=5.0, partial=False, seed=0, front="car"):
    """A (car, moving down at `speed` px/frame) is hidden by B (a bigger vehicle passing in front, same direction, slower->overtakes) for occ_len frames starting at frame occ_start.
    While hidden A is NOT detected (partial=True: A is detected but only as a small visible sliver with a shifted centre)."""
    random.seed(seed); trk = N.VehicleTracker([(0.0, 0.5), (1.0, 0.5)], False, "center", 3, 8, groups=G); ev = []; fid = 0; ya = 120.0; yb = 180.0
    for k in range(260):
        fid += 1; ya += speed; dets = []
        occluded = occ_start <= k < occ_start + occ_len
        # B: truck passing in front of A during the occlusion window, otherwise far away (not in the picture)
        if occluded or (occ_start - 8 <= k < occ_start) or (occ_start + occ_len <= k < occ_start + occ_len + 8):
            yb = ya + 5 if occluded else ya + (60 if k < occ_start else -60) * 1.0
            dets.append((box(330, yb, 240, 200), 0.9, 4))                                                # B: large vehicle, group 2
        if not occluded: dets.append((box(330 + random.uniform(-3, 3), ya, 150, 120), 0.8, 3))          # A visible
        elif partial: dets.append((box(330 + 40, ya + 30, 60, 50), 0.5, 3))                              # only a sliver of A visible, centre shifted
        ev += trk.update([d[0] for d in dets], [d[1] for d in dets], [d[2] for d in dets], W, H, fid, fid / 20.0)[1]
    return [N.TRAFFIC_CLASSES[e[1]] if e[1] < len(N.TRAFFIC_CLASSES) else e[1] for e in ev]
print("A drives across the line (reaches it at frame ~34); B (truck) hides it for N frames. Expected: A counted exactly once.")
rows = (("hidden BEFORE the line", lambda n: 34 - n - 5), ("hidden OVER the crossing", lambda n: 34 - n // 2), ("hidden just AFTER the line", lambda n: 38))
tot = bad = 0
for label, st in rows:
    row = []
    for n in (6, 10, 20, 30, 45):
        res = [run(st(n), n, seed=s) for s in range(4)]; carc = [r.count("car") for r in res]; tot += 4; bad += sum(1 for c in carc if c != 1); row.append("%2d fr: %s" % (n, "".join(str(c) for c in carc)))
    print("%-27s %s" % (label, " | ".join(row)))
print("A counted wrongly (0 = missed, 2 = double): %d of %d runs" % (bad, tot))
print("\nboth vehicles: all events (A car + B truck that overtakes and crosses), hidden over the crossing:")
for n in (10, 30, 45): print("  %2d frames hidden ->" % n, run(34 - n // 2, n, seed=1))
print("partial occlusion (only a small sliver of A visible, shifted centre) hidden over the crossing:")
bad = 0
for n in (10, 20, 30, 45):
    r = [run(34 - n // 2, n, partial=True, seed=s).count("car") for s in range(4)]; bad += sum(1 for c in r if c != 1); print("  %2d fr: %s" % (n, "".join(map(str, r))))
print("partial: wrong in %d of 16 runs" % bad)
