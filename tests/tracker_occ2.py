import sys, importlib.util, random
spec = importlib.util.spec_from_file_location("N", sys.argv[1]); N = importlib.util.module_from_spec(spec); sys.modules["N"] = N; spec.loader.exec_module(N)
W, H = 704, 576; Y0 = 0.5 * H; G = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
def box(cx, cy, w, h): return (int(cx - w / 2), int(cy - h / 2), w, h)
def run(seed):
    random.seed(seed); trk = N.VehicleTracker([(0.0, 0.5), (1.0, 0.5)], False, "center", 3, 8, groups=G); ev = []; ya = 120.0
    for k in range(200):
        ya += 5; d = []; hidden = 24 <= k < 54
        d.append((box(330, ya + 5, 240, 200), 0.9, 4)) if hidden else d.append((box(330 + random.uniform(-3, 3), ya, 150, 120), 0.8, 3))      # car A hidden by a big vehicle for 30 frames over the line
        if 30 <= k < 90: d.append((box(335, 120 + 6 * (k - 30), 50, 90), 0.7, 1))                                                                  # a SCOOTER drives across the line, in front of / inside the hidden car's box
        ev += trk.update([x[0] for x in d], [x[1] for x in d], [x[2] for x in d], W, H, k + 1, (k + 1) / 20.0)[1]
    return sorted(N.TRAFFIC_CLASSES[e[1]] for e in ev)
print("car hidden over the line + scooter crossing in front of it:"); [print("  seed", s, run(s)) for s in range(3)]
