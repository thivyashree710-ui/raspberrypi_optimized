import sys, random, importlib.util
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
N = load(sys.argv[1], "N"); W, H = 704, 576; Y0 = 0.5 * H; GROUPS = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
def box(cx, cy, w, h): return (int(cx - w / 2), int(cy - h / 2), w, h)
def run(wander, seed, big=True, creep=0.0):
    random.seed(seed); trk = N.VehicleTracker([(0.0, 0.5), (1.0, 0.5)], False, "center", 3, 8, groups=GROUPS); ev = []; fid = 0; y = 150.0; w, h = (300, 230) if big else (90, 70)
    def feed(d):
        nonlocal fid; fid += 1; return trk.update([x[0] for x in d], [x[1] for x in d], [x[2] for x in d], W, H, fid, fid / 15.0)[1]
    while y < Y0 + 60: ev += feed([(box(350, y, w, h), 0.9, 3)]); y += 12                      # drives in and crosses clearly
    off = 0.0
    for step in range(15 * 90):                                                                  # then stands / creeps 90 s, box wandering, scooters hiding it for 12-30 frames every few seconds
        if step % 70 < random.choice((12, 20, 30)) and step % 70 > 55: ev += feed([]); continue
        off += random.uniform(-3, 3); off = max(-wander, min(wander, off))
        if creep: off += creep * __import__('math').sin(step / 40.0)                       # stop-and-go: rolls forward and back a little all the time                         # slow wander of the box centre
        ev += feed([(box(350 + random.uniform(-10, 10), Y0 + 60 + off + random.uniform(-8, 8), w + random.uniform(-25, 25), h + random.uniform(-20, 20)), random.uniform(0.5, 0.95), 3)])
    return len(ev)
bad = 0; tot = 0
for creep in (0.0, 2.5, 5.0):
    for wander in (45, 90):
        for seed in range(6):
            tot += 1; n = run(wander, seed, creep=creep); bad += n != 1
            if n != 1: print("  FAIL creep %.1f wander %d seed %d: counted %d times" % (creep, wander, seed, n))
print("standing/creeping car counted exactly once: %d of %d scenarios OK" % (tot - bad, tot))
