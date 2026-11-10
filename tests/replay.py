import sys, pickle, math, json, importlib.util, collections
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
NEW = load("/home/cosai/New_optimized/New_optimized.py", "N_new"); OLD = load("/home/cosai/New_optimized/New_optimized.before_tracker2.py", "N_old")
names = NEW.load_names("/home/cosai/New_optimized/yolo_256_int8_xnnpack.pte"); line = NEW.load_line("count_line.json")
groups = {i: (0 if n.lower() in ("two_wheeler", "cycle", "person") else 1 if n.lower() == "auto_rickshaw" else 2) for i, n in enumerate(names)}
def run(mod, D, new):
    w, h = D["w"], D["h"]; trk = mod.VehicleTracker(line, False, "center", 3, 8, groups=groups) if new else mod.VehicleTracker(line, False, "center", 3, 8)
    ev = []; last = {}; jumps = 0; gchg = 0; steps = 0; pos = {}
    for fid, (b, s, c) in enumerate(D["dets"], 1):
        ids, e = (trk.update(b, s, c, w, h, fid, fid / 25.0) if new else trk.update(b, s, c, w, h, fid)); ev += [(x, fid) for x in e]
        for (x, y, bw, bh), tid, cc in zip(b, ids, c):
            cx, cy = x + bw / 2, y + bh / 2
            if tid in last:
                px, py, pw, ph, pc, pf = last[tid]; steps += 1
                if fid - pf <= 3 and math.hypot(cx - px, cy - py) > 0.8 * max(pw, ph): jumps += 1                      # one id jumped to a spot that a vehicle of that size could not have reached
                if groups.get(cc, cc) != groups.get(pc, pc) and fid - pf <= 3: gchg += 1
            last[tid] = (cx, cy, bw, bh, cc, fid)
        for (tid, cls, d, f) in e: pos[(tid, f)] = trk.info.get(tid)
    rows = trk.done + [(t["id"], t["first"], t["last"], t["hits"], 0, 0, t["cross"] or "", t["counted"], t["note"]) for t in trk.tracks]
    dup = 0; evs = [(x[0], x[1], x[2], x[3], pos.get((x[0], x[3]))) for x, f in ev]
    for i in range(len(evs)):
        for j in range(i + 1, len(evs)):
            a, b2 = evs[i], evs[j]
            if abs(a[3] - b2[3]) <= 125 and a[4] and b2[4] and groups.get(a[1], a[1]) == groups.get(b2[1], b2[1]):
                (ax, ay, aw, ah), (bx, by, bw, bh) = a[4][1], b2[4][1]; 
                if math.hypot(ax + aw / 2 - bx - bw / 2, ay + ah / 2 - by - bh / 2) < 0.8 * max(aw, ah, bw, bh): dup += 1
    cnt = collections.Counter((names[e[1]], e[2]) for e, f in ev)
    return {"events": len(ev), "in": sum(1 for e, f in ev if e[2] == "IN"), "out": sum(1 for e, f in ev if e[2] == "OUT"), "tracks": len(rows), "id_jump_steps": jumps, "group_change_steps": gchg, "tracked_steps": steps, "dup_suspects": dup, "by_class": dict(cnt)}, [(e[0], names[e[1]], e[2], f) for e, f in ev]
for tag in ("d128", "d256"):
    D = pickle.load(open(tag + ".pkl", "rb")); print("=====", tag, "frames", len(D["dets"]), "line", line)
    res = {}
    for label, mod, new in (("old tracker", OLD, False), ("new tracker", NEW, True)):
        r, ev = run(mod, D, new); res[label] = (r, ev); print("%-12s events %-3d (IN %d / OUT %d) | tracks %-3d | id-jump steps %-2d of %d | class-group changes %d | duplicate-count suspects %d" % (label, r["events"], r["in"], r["out"], r["tracks"], r["id_jump_steps"], r["tracked_steps"], r["group_change_steps"], r["dup_suspects"]))
    print("   by class (new):", res["new tracker"][0]["by_class"]); print("   by class (old):", res["old tracker"][0]["by_class"])
    o = {(e[0]) for e in res["old tracker"][1]}; n = {(e[0]) for e in res["new tracker"][1]}
    pickle.dump({k: v[1] for k, v in res.items()}, open("events_" + tag + ".pkl", "wb"))
