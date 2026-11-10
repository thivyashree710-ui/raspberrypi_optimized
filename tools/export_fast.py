"""Re-export the 18-class YOLO11s core (6 raw heads) to an ExecuTorch XNNPACK .pte with FEWER graph breaks.  Same pipeline as the user's model_converter_code.py (SiLU -> x*sigmoid(x), raw head outputs,
PT2E static INT8, symmetric per-channel XNNPACKQuantizer, XnnpackPartitioner, same EdgeCompileConfig) - the ONLY differences are mathematically identical rewrites that XNNPACK can run:
   chunk/split  ->  slicing              (C2f/C3k2 chunk(2,1), C2PSA split, Attention split)      [variant 'splits']
   nearest x2 upsample -> depthwise ConvTranspose2d (kernel 2, stride 2, weights 1)               [variant 'all']
usage: export_fast.py VARIANT PRECISION OUT_NAME      VARIANT = orig | splits | all      PRECISION = fp32 | int8
Nothing outside this folder is modified. The original model file and checkpoint are only read."""
import sys, os, copy, glob, json, collections, hashlib
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); VARIANT, PREC, NAME = sys.argv[1], sys.argv[2], sys.argv[3]; S = int(os.environ.get("IMGSZ", 256))
CKPT = "/home/cosai/Downloads/18cls_9L_last_saved_model-1/18cls_9L_last_saved_model/18cls_9L_last_recovered.pt"
from ultralytics import YOLO
from ultralytics.nn.modules import block
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_transform_passes
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class XNNPACKSiLU(nn.Module):                                   # identical to the user's script
    def forward(self, x): return x * torch.sigmoid(x)


def replace_silu(m):
    for n, c in m.named_children():
        if isinstance(c, nn.SiLU): setattr(m, n, XNNPACKSiLU())
        else: replace_silu(c)


class Core(nn.Module):                                          # identical to the user's QuantizableYoloCore
    def __init__(self, y):
        super().__init__(); self.layers = y.model[:-1]; self.save = y.save; d = y.model[-1]; self.detect_f = d.f; self.cv2 = d.cv2; self.cv3 = d.cv3
    def forward(self, x):
        y = []
        for m in self.layers:
            if m.f != -1: x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x); y.append(x if m.i in self.save else None)
        f = [x if j == -1 else y[j] for j in self.detect_f]
        return self.cv2[0](f[0]), self.cv2[1](f[1]), self.cv2[2](f[2]), self.cv3[0](f[0]), self.cv3[1](f[1]), self.cv3[2](f[2])


class Up2x(nn.Module):
    """nearest-neighbour x2 upsample as a depthwise transposed convolution (kernel 2, stride 2, all-ones weights): exactly the same values."""
    def __init__(self, c):
        super().__init__(); self.deconv = nn.ConvTranspose2d(c, c, 2, 2, groups=c, bias=False)
        with torch.no_grad(): self.deconv.weight.fill_(1.0)
        self.deconv.weight.requires_grad_(False)
    def forward(self, x): return self.deconv(x)


class Up2xCat(nn.Module):
    """nearest x2 upsample using only unsqueeze + cat + reshape (exactly the same values): [x0,x0,x1,x1,...] along width, then along height."""
    def forward(self, x):
        B, C, H, W = x.shape
        y = torch.cat([x.unsqueeze(-1), x.unsqueeze(-1)], dim=-1).reshape(B, C, H, 2 * W)
        return torch.cat([y.unsqueeze(3), y.unsqueeze(3)], dim=3).reshape(B, C, 2 * H, 2 * W)


def patch_splits():
    def c2f_forward(self, x):
        t = self.cv1(x); y = [t[:, :self.c], t[:, self.c:]]; y.extend(m(y[-1]) for m in self.m); return self.cv2(torch.cat(y, 1))
    def c2psa_forward(self, x):
        t = self.cv1(x); a, b = t[:, :self.c], t[:, self.c:]; b = self.m(b); return self.cv2(torch.cat((a, b), 1))
    def attn_forward(self, x):
        B, C, H, W = x.shape; N = H * W; t = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N); kd = self.key_dim
        q, k, v = t[:, :, :kd], t[:, :, kd:2 * kd], t[:, :, 2 * kd:]
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1); x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W)); return self.proj(x)
    block.C2f.forward = c2f_forward; block.C2PSA.forward = c2psa_forward; block.Attention.forward = attn_forward


def load_core(variant):
    y = YOLO(CKPT); base = copy.deepcopy(y.model).to("cpu").eval(); replace_silu(base)
    for p in base.parameters(): p.requires_grad = False
    if variant in ("all", "cat"):                                        # find the channel count at each Upsample with one dry run, then swap them
        ch = {}; hooks = [m.register_forward_hook(lambda m_, i, o, n_=n: ch.__setitem__(n_, i[0].shape[1])) for n, m in base.named_modules() if isinstance(m, nn.Upsample)]
        with torch.no_grad(): Core(base).eval()(torch.zeros(1, 3, S, S))
        [h.remove() for h in hooks]
        for n, c in ch.items():
            idx = int(n.split(".")[1]); old = base.model[idx]; up = Up2x(c) if variant == "all" else Up2xCat(); up.f, up.i, up.type, up.np = old.f, old.i, "Up2x", 0; base.model[idx] = up
    return Core(base).eval()


def stats(buf):
    plan = deserialize_pte_binary(bytes(buf)).program.execution_plan[0]; ops = [o.name for o in plan.operators]; kc = collections.Counter(); nd = 0
    for ch in plan.chains:
        for ins in ch.instructions:
            a = ins.instr_args
            if type(a).__name__ == "DelegateCall": nd += 1
            elif type(a).__name__ == "KernelCall": kc[ops[a.op_index]] += 1
    return {"xnnpack_partitions": nd, "portable_calls": sum(kc.values()), "portable_ops": dict(kc)}


def preprocess(path):
    img = cv2.imread(path); img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).contiguous()


if __name__ == "__main__":
    imgs = sorted(glob.glob(f"{HERE}/calib_cam/*.jpg")); cal, held = imgs[:50], imgs[50:]
    if os.environ.get("CALIB_DIR"): cal = sorted(glob.glob(os.environ["CALIB_DIR"] + "/*.jpg"))      # a different calibration set (never overlaps the evaluation frames)
    ref_in = [preprocess(p) for p in imgs[-6:]]
    with torch.no_grad(): ref_out = [load_core("orig")(x) for x in ref_in]          # reference from the UNPATCHED model, computed before any patching
    if VARIANT in ("splits", "all", "cat"): patch_splits()
    core = load_core(VARIANT); sample = (torch.randn(1, 3, S, S),)
    with torch.no_grad(): new_out = [core(x) for x in ref_in]
    worst = max(float((a - b).abs().max()) for ra, rb in zip(ref_out, new_out) for a, b in zip(ra, rb)); print(f"[{VARIANT}] FP32 equivalence vs the original model: max abs difference over 6 outputs x {len(ref_in)} frames = {worst:.2e}", flush=True)
    if PREC == "fp32": graph = torch.export.export(core, sample, strict=False)
    else:
        q = XNNPACKQuantizer(); q.set_global(get_symmetric_quantization_config(is_per_channel=True)); prep = prepare_pt2e(torch.export.export(core, sample, strict=False).module(), q)
        with torch.no_grad():
            for p in cal: prep(preprocess(p))
        graph = torch.export.export(convert_pt2e(prep), sample, strict=False)
    em = to_edge_transform_and_lower(graph, partitioner=[XnnpackPartitioner()], transform_passes=get_transform_passes(), compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True))
    buf = em.to_executorch().buffer; out = f"{HERE}/{NAME}.pte"; open(out, "wb").write(buf); st = stats(buf)
    st.update({"variant": VARIANT, "precision": PREC, "file": out, "bytes": len(buf), "sha256_12": hashlib.sha256(buf).hexdigest()[:12], "fp32_equiv_max_abs": worst, "calib_images": len(cal) if PREC == "int8" else 0})
    json.dump(st, open(f"{HERE}/{NAME}.json", "w"), indent=1); print(json.dumps({k: st[k] for k in ("variant", "precision", "bytes", "xnnpack_partitions", "portable_calls", "portable_ops", "fp32_equiv_max_abs")}), flush=True)
