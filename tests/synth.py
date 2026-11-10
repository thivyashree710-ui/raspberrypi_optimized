import sys, random, importlib.util, math
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
which = sys.argv[1]; N = load("/home/cosai/New_optimized/New_optimized.py" if which == "new" else "/home/cosai/New_optimized/New_optimized.before_tracker2.py", "N_" + which)
W, H = 704, 576; LINE = [(0.0, 0.5), (1.0, 0.5)]; Y0 = 0.5 * H; random.seed(7); R = []
GROUPS = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}                      # 0 two-wheeler, 1 cycle, 2 auto, 3 car, 4 van
def mk(kw=None): return N.VehicleTracker(LINE, False, "center", 3, 8, groups=GROUPS, **(kw or {})) if which == "new" else N.VehicleTracker(LINE, False, "center", 3, 8)
def feed(trk, dets, fid, hz=15.0):
    b = [d[0] for d in dets]; s = [d[1] for d in dets]; c = [d[2] for d in dets]
    return trk.update(b, s, c, W, H, fid, fid / hz) if which == "new" else trk.update(b, s, c, W, H, fid)
def box(cx, cy, w=90, h=70): return (int(cx - w / 2), int(cy - h / 2), w, h)
def report(name, exp, got, extra=""): R.append((name, exp, got)); print("%-4s %-62s expected %-3s got %-3s %s" % ("OK" if exp == got else "FAIL", name, exp, got, extra))

# T1: crosses CLEARLY, backs up onto the line and PARKS there for 10 minutes: jitter +-24 px (flips either side), 12% random misses, blackout every minute
for dropout_s, label in ((0, "no blackout"), (40, "40 s blackout every minute"), (-1, "one 4-minute blackout")):
    trk = mk(); ev = []; fid = 0; y = 200.0
    while y < Y0 + 40: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 0)], fid)[1]; y += 14      # approach and cross clearly
    while y > Y0: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 0)], fid)[1]; y -= 10           # back onto the line
    for step in range(15 * 600):                                                                  # 10 minutes at 15 detections/s
        fid += 1; t = step / 15.0
        if dropout_s > 0 and t >= 60 and (t % 60) < dropout_s: ev += feed(trk, [], fid)[1]; continue
        if dropout_s < 0 and 120 <= t < 360: ev += feed(trk, [], fid)[1]; continue
        if random.random() < 0.12: ev += feed(trk, [], fid)[1]; continue
        n0 = len(ev); ev += feed(trk, [(box(300 + random.uniform(-6, 6), Y0 + random.uniform(-24, 24)), random.uniform(0.4, 0.9), 0 if random.random() > 0.2 else 1)], fid)[1]
        if which == "new" and dropout_s == 40 and len(ev) > n0 and n0 >= 1 and not globals().get("_p"):
            globals()["_p"] = 1; print("   DEBUG t=%.1f events=%d tracks=%s mem=%s" % (t, len(ev), [(x["id"], x["cross"], x["counted"], x["side"], len(x["hist"]), round(trk._radius(x)), x["note"][:18]) for x in trk.tracks], [(e["id"], e["counted"], round(e["cx"]), round(e["cy"]), round(e["t"], 1)) for e in trk.mem]), "done:", [(r[0], r[2], r[3], r[8][:28]) for r in trk.done[-3:]])
    report("T1 crosses, parks ON the line 10 min (%s)" % label, 1, len(ev))

# T2: drives in and stops exactly ON the line (never clearly crossed), 10 minutes with dropouts, then drives away across it
trk = mk(); ev = []; fid = 0; y = 200.0
while y < Y0 - 4: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y += 10
for step in range(15 * 600):
    fid += 1
    if random.random() < 0.12 or ((step / 15.0) >= 60 and (step / 15.0) % 90 < 35): ev += feed(trk, [], fid)[1]; continue
    ev += feed(trk, [(box(300 + random.uniform(-5, 5), Y0 + random.uniform(-9, 9)), 0.8, 3)], fid)[1]
n_parked = len(ev)
while y < 420: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y += 14
report("T2 parked in the line's dead-band 10 min: nothing counted while parked", 0, n_parked)
report("T2 ... then drives away across the line: counted exactly once", 1, len(ev))

# T3: two vehicles side by side move across together, one of them missed for 3 detections in the middle; ids must stay with their own vehicle
trk = mk(); ev = []; fid = 0; ids_a = set(); ids_b = set(); y = 150.0
while y < 420:
    fid += 1; dets = [(box(250, y, 80, 60), 0.9, 0), (box(330, y + 4, 90, 70), 0.9, 3)]
    if 20 <= (fid % 60) < 23: dets = dets[:1]
    ids, e = feed(trk, dets, fid); ev += e; ids_a.add(ids[0]); 
    if len(ids) > 1: ids_b.add(ids[1])
    y += 12
report("T3 two vehicles side by side: both counted once", 2, len(ev), "events=%s" % [(e[0], e[2]) for e in ev])
report("T3 ... each vehicle keeps ONE id (no id switch)", (1, 1), (len(ids_a), len(ids_b)))

# T4: crosses IN, then reverses back across the line: still ONE count
trk = mk(); ev = []; fid = 0; y = 200.0
while y < 360: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y += 12
while y > 200: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y -= 12
report("T4 crosses then reverses back over the line: counted once", 1, len(ev))

# T5: class flicker car <-> van while crossing: one count
trk = mk(); ev = []; fid = 0; y = 200.0
while y < 380: fid += 1; ev += feed(trk, [(box(300, y), 0.7, 3 if fid % 2 else 4)], fid)[1]; y += 12
report("T5 class flickers car/van while crossing: counted once", 1, len(ev))

# T6: a different vehicle later parks on the same spot, after the first one left more than the memory time ago: counted separately
trk = mk(); ev = []; fid = 0; y = 200.0
while y < 380: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y += 12
for _ in range(15 * 400): fid += 1; ev += feed(trk, [], fid)[1]                               # empty road for 400 s
y = 200.0
while y < 380: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y += 12
report("T6 two different vehicles 400 s apart: both counted", 2, len(ev))
# T7: first vehicle crosses and parks ON the line (counted once), drives away; 30 s later a DIFFERENT vehicle drives in, crosses and parks on the same spot: 2 vehicles in total
trk = mk(); ev = []; fid = 0; y = 200.0
while y < Y0 + 40: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y += 14
while y > Y0: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y -= 10
for step in range(15 * 200):
    fid += 1
    if random.random() < 0.12 or ((step / 15.0) >= 60 and (step / 15.0) % 60 < 40): ev += feed(trk, [], fid)[1]; continue
    ev += feed(trk, [(box(300 + random.uniform(-6, 6), Y0 + random.uniform(-24, 24)), 0.8, 3)], fid)[1]
n1 = len(ev)
while y < 520: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y += 14                 # first vehicle drives away
for _ in range(15 * 30): fid += 1; ev += feed(trk, [], fid)[1]
y = 200.0
while y < Y0 + 40: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y += 14             # a second vehicle arrives
while y > Y0: fid += 1; ev += feed(trk, [(box(300, y), 0.9, 3)], fid)[1]; y -= 10
for step in range(15 * 200):
    fid += 1
    if random.random() < 0.12 or ((step / 15.0) >= 60 and (step / 15.0) % 60 < 40): ev += feed(trk, [], fid)[1]; continue
    ev += feed(trk, [(box(300 + random.uniform(-6, 6), Y0 + random.uniform(-24, 24)), 0.8, 3)], fid)[1]
report("T7 first vehicle counted once while parked", 1, n1)
report("T7 a different vehicle parks on the same spot later: counted as 2nd", 2, len(ev))

print("\nRESULT:", "ALL PASS" if all(e == g for _, e, g in R) else "%d FAIL" % sum(e != g for _, e, g in R))
