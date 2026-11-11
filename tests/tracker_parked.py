import sys, random, importlib.util
spec = importlib.util.spec_from_file_location("N", sys.argv[1]); N = importlib.util.module_from_spec(spec); sys.modules["N"] = N; spec.loader.exec_module(N)
W, H = 704, 576; Y0 = 0.5 * H; G = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
def box(cx, cy, w, h): return (int(cx - w / 2), int(cy - h / 2), w, h)
def run(path_fn, n, w, h, cls, seed):
    random.seed(seed); trk = N.VehicleTracker([(0.0, 0.5), (1.0, 0.5)], False, "center", 3, 8, groups=G); ev = []
    for fid in range(1, n + 1):
        cx, cy = path_fn(fid); ev += trk.update([box(cx, cy, w + random.uniform(-6, 6), h + random.uniform(-6, 6))], [0.6], [cls], W, H, fid, fid / 20.0)[1]
    ev += trk.flush(n + 1) if hasattr(trk, "flush") else []
    return len(ev)
def parked(seed):                       # sits on the spot, rider moves about: centre wanders +-35 px, crossing the dead-band back and forth
    off = [0.0, 0.0]
    def f(fid):
        off[0] = max(-25, min(25, off[0] + random.uniform(-9, 9))); off[1] = max(-35, min(35, off[1] + random.uniform(-9, 9))); return 330 + off[0], Y0 + off[1]
    return f
fast = lambda fid: (330, 150 + 8 * fid)                                   # normal traffic, 8 px/frame
creep = lambda fid: (330 + random.uniform(-3, 3), 150 + 1.6 * fid + random.uniform(-3, 3))      # slow queue crawl with noisy box
cases = (("parked scooter, rider fidgets (expect 0)", lambda s: run(parked(s), 140, 70, 110, 1, s), 0), ("normal fast vehicle (expect 1)", lambda s: run(fast, 70, 70, 110, 1, s), 1), ("slow noisy creeping car (expect 1)", lambda s: run(creep, 210, 160, 120, 3, s), 1))
for name, fn, exp in cases:
    got = [fn(s) for s in range(10)]; print("%-42s counts over 10 runs: %s  -> %s" % (name, got, "OK" if all(g == exp for g in got) else "(not all %d)" % exp))
