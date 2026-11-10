import sys, pickle, time, cv2, numpy as np, torch
sys.path.insert(0, "/home/cosai/New_optimized"); import New_optimized as N
from executorch.extension.pybindings import portable_lib as pl
pte, S, clip, out, zf = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
zones = N.load_zones(zf); mod = pl._load_for_executorch(pte); pre = N.PreProc(S, False); buf = np.zeros((1, 3, S, S), np.float32); t = torch.from_numpy(buf); cap = cv2.VideoCapture(clip); dets = []; t0 = time.time()
while True:
    ok, f = cap.read()
    if not ok: break
    h, w = f.shape[:2]; geo = pre(f, buf); outs = [o.numpy() for o in mod.forward([t])]; b, s, c = N.decode_heads(outs, w, h, S, 0.35, 0.45, geo, False, zones, "center"); dets.append(([tuple(int(v) for v in x) for x in b], [float(v) for v in s], [int(v) for v in c]))
pickle.dump({"dets": dets, "w": w, "h": h, "names": N.load_names(pte) if False else None}, open(out, "wb")); print(out, len(dets), "frames in %.0f s" % (time.time() - t0))
