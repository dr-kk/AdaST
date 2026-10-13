#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaST E3: Content-Adaptive Token Budget for TimeSformer / UCF101.

E2-v2 fixed a 50% per-frame budget. E3 learns a per-video retention ratio
within [r_min, r_max], while preserving per-frame temporal coverage.

IMPORTANT:
- This experiment uses hard shape-preserving gating, not physical token deletion.
- Therefore latency/FPS are diagnostic only and must not be claimed as physical
  token-pruning acceleration.
"""

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification

MODEL_NAME = "facebook/timesformer-base-finetuned-k400"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_index(split_root):
    p = Path(split_root) / "classInd.txt"
    if not p.exists():
        raise FileNotFoundError(p)
    m = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            i, c = line.split(maxsplit=1)
            m[c] = int(i) - 1
    return m


def read_split(data_root, split_root, split):
    data_root, split_root = Path(data_root), Path(split_root)
    cmap = class_index(split_root)
    trf = split_root / f"trainlist{split:02d}.txt"
    tef = split_root / f"testlist{split:02d}.txt"
    if not trf.exists() or not tef.exists():
        raise FileNotFoundError(f"Missing split files under {split_root}")

    train = []
    for line in trf.read_text(encoding="utf-8").splitlines():
        z = line.strip().split()
        if z:
            rel = z[0].replace("/", os.sep)
            train.append((data_root / rel, cmap[Path(rel).parts[0]]))

    test = []
    for line in tef.read_text(encoding="utf-8").splitlines():
        rel = line.strip().replace("/", os.sep)
        if rel:
            test.append((data_root / rel, cmap[Path(rel).parts[0]]))
    return train, test


def stratified_split(items, val_fraction, seed):
    groups = defaultdict(list)
    for x in items:
        groups[x[1]].append(x)
    rng = random.Random(seed)
    tr, va = [], []
    for y in sorted(groups):
        g = groups[y][:]
        rng.shuffle(g)
        nv = max(1, round(len(g) * val_fraction))
        nv = min(nv, len(g) - 1)
        va += g[:nv]
        tr += g[nv:]
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def decode_video(path, frames):
    try:
        c = av.open(str(path))
    except Exception as e:
        raise RuntimeError(f"Cannot open {path}: {e}") from e
    xs = []
    try:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for fr in c.decode(stream):
            xs.append(fr.to_ndarray(format="rgb24"))
    finally:
        c.close()
    if not xs:
        raise RuntimeError(f"No frames decoded: {path}")
    ids = np.linspace(0, len(xs) - 1, frames).round().astype(int)
    return [xs[i] for i in ids]


class UCF(Dataset):
    def __init__(self, items, processor, frames):
        self.items, self.processor, self.frames = items, processor, frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        ims = decode_video(p, self.frames)
        x = self.processor(images=ims, return_tensors="pt")["pixel_values"].squeeze(0)
        return x, int(y), str(p)


def topk_hits(logits, y):
    _, p = logits.topk(min(5, logits.size(1)), 1)
    hit1 = p[:, :1].eq(y[:, None]).sum().item()
    hit5 = p.eq(y[:, None]).any(dim=1).sum().item()
    return hit1, hit5


class ImportanceScorer(nn.Module):
    def __init__(self, d=768, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3*d, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, 1)
        )

    def forward(self, patch, T):
        b, m, d = patch.shape
        n = m // T
        z = patch.reshape(b, T, n, d)
        delta = torch.zeros_like(z)
        delta[:, 1:] = z[:, 1:] - z[:, :-1]
        ctx = z.mean((1, 2), keepdim=True).expand(-1, T, n, -1)
        feat = torch.cat([z, delta, ctx], -1)
        s = torch.sigmoid(self.net(feat)).squeeze(-1)
        return s.reshape(b, m), delta


class BudgetNet(nn.Module):
    def __init__(self, d=768, h=128):
        super().__init__()
        self.cls = nn.Sequential(nn.Linear(d, 64), nn.GELU())
        self.net = nn.Sequential(
            nn.Linear(67, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, 1)
        )
        # Small non-zero initialization is essential: zero weights would make
        # all clips start identically and delay content differentiation.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, scores, delta, cls, rmin, rmax):
        sm = scores.mean(1, keepdim=True)
        ss = scores.std(1, keepdim=True, unbiased=False)
        tm = delta.abs().mean((1, 2, 3)).unsqueeze(1)
        cf = self.cls(cls.squeeze(1))
        q = torch.cat([sm, ss, tm, cf], 1)
        raw = self.net(q).squeeze(1)
        return rmin + (rmax-rmin) * torch.sigmoid(raw)


class AdaSTE3(nn.Module):
    def __init__(self, backbone, block, T, rmin, rmax, scorer_h=256, budget_h=128):
        super().__init__()
        self.backbone = backbone
        self.block, self.T = block, T
        self.rmin, self.rmax = rmin, rmax
        d = backbone.config.hidden_size
        self.scorer = ImportanceScorer(d, scorer_h)
        self.budget = BudgetNet(d, budget_h)
        self.last_r = self.last_actual_r = self.last_k = None
        self.last_scores = None
        self.last_n = None
        layers = self.layers()
        self.hook = layers[block].register_forward_pre_hook(
            self.gate_hook, with_kwargs=True
        )

    def layers(self):
        return self.backbone.timesformer.encoder.layer

    def gate_hook(self, module, args, kwargs):
        if args:
            h, rest, src = args[0], args[1:], "args"
        else:
            h, rest, src = kwargs["hidden_states"], (), "kwargs"

        cls, patch = h[:, :1], h[:, 1:]
        b, m, d = patch.shape
        if m % self.T:
            raise RuntimeError(f"{m} patch tokens not divisible by T={self.T}")
        n = m // self.T
        scores, delta = self.scorer(patch, self.T)
        r = self.budget(scores, delta, cls, self.rmin, self.rmax)
        k = torch.round(r * n).long().clamp(1, n)

        sf = scores.reshape(b, self.T, n)
        maskf = torch.zeros_like(sf)
        for bi in range(b):
            ki = int(k[bi].item())
            ids = torch.topk(sf[bi], ki, dim=1, largest=True, sorted=False).indices
            maskf[bi].scatter_(1, ids, 1.0)
        hard = maskf.reshape(b, m)

        # Straight-through importance gate plus budget gradient surrogate.
        gate = (hard + scores - scores.detach())
        budget_st = 1.0 + r - r.detach()
        gate = gate * budget_st[:, None]

        gh = torch.cat([cls, patch * gate.unsqueeze(-1)], 1)
        self.last_r = r
        self.last_actual_r = hard.mean(1).detach()
        self.last_k = k.detach()
        self.last_scores = scores
        self.last_n = n

        if src == "args":
            return (gh,) + rest, kwargs
        kw = dict(kwargs)
        kw["hidden_states"] = gh
        return args, kw

    def forward(self, x):
        return self.backbone(pixel_values=x, return_dict=True)


def state_dict_from(obj):
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise RuntimeError("Unsupported checkpoint")


def normalize(sd):
    out = {}
    for k, v in sd.items():
        while k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def load_dense(backbone, path):
    sd = normalize(state_dict_from(torch.load(path, map_location="cpu", weights_only=False)))
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[9:] if k.startswith("backbone.") else k: v for k, v in sd.items()}
    z = backbone.load_state_dict(sd, strict=False)
    print(f"Dense checkpoint: missing={len(z.missing_keys)}, unexpected={len(z.unexpected_keys)}")


def load_e2(model, path):
    sd = normalize(state_dict_from(torch.load(path, map_location="cpu", weights_only=False)))
    z = model.load_state_dict(sd, strict=False)
    bad = [k for k in z.missing_keys if not k.startswith("budget.")]
    print(f"E2-v2 init: missing={len(z.missing_keys)}, unexpected={len(z.unexpected_keys)}")
    if bad:
        raise RuntimeError("Incompatible E2 checkpoint; non-budget missing keys: " + str(bad[:10]))


def configure(model, start):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.scorer.parameters():
        p.requires_grad = True
    for p in model.budget.parameters():
        p.requires_grad = True
    for i in range(start, len(model.layers())):
        for p in model.layers()[i].parameters():
            p.requires_grad = True
    for p in model.backbone.classifier.parameters():
        p.requires_grad = True
    for p in model.backbone.timesformer.layernorm.parameters():
        p.requires_grad = True


def run_epoch(model, loader, device, amp, optimizer=None, scaler=None,
              accum=1, target=.5, lam_budget=5.0):
    training = optimizer is not None
    model.train(training)
    ce = nn.CrossEntropyLoss()
    total = c1 = c5 = 0
    loss_sum = 0.0
    rs, rhs, ks = [], [], []

    if training:
        optimizer.zero_grad(set_to_none=True)

    ctx = torch.enable_grad if training else torch.no_grad
    with ctx():
        for step, (x, y, _) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=(amp and device.type == "cuda")):
                o = model(x)
                cls_loss = ce(o.logits, y)
                # Per-sample budget centering. This keeps average compute near
                # target while classification gradients can differentiate clips.
                budget_loss = (model.last_r.mean() - target).pow(2)
                loss = cls_loss + lam_budget * budget_loss

            if training:
                bl = loss / accum
                if scaler:
                    scaler.scale(bl).backward()
                else:
                    bl.backward()
                if step % accum == 0 or step == len(loader):
                    pars = [p for p in model.parameters() if p.requires_grad]
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(pars, 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(pars, 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            a, b = topk_hits(o.logits.detach(), y)
            bs = y.size(0)
            total += bs; c1 += a; c5 += b
            loss_sum += loss.item() * bs
            rs += model.last_r.detach().float().cpu().tolist()
            rhs += model.last_actual_r.float().cpu().tolist()
            ks += model.last_k.cpu().tolist()

            if training and (step % 200 == 0 or step == len(loader)):
                ar = np.asarray(rs)
                print(f"  batch {step}/{len(loader)} | loss {loss_sum/total:.4f} "
                      f"| top1 {100*c1/total:.2f}% | rho {ar.mean():.4f} "
                      f"| rho_std {ar.std():.4f}")

    r = np.asarray(rs); rh = np.asarray(rhs); k = np.asarray(ks)
    return dict(
        loss=loss_sum/max(total,1), top1=100*c1/max(total,1),
        top5=100*c5/max(total,1), r_mean=float(r.mean()),
        r_std=float(r.std()), r_min=float(r.min()), r_max=float(r.max()),
        actual_r_mean=float(rh.mean()), actual_r_std=float(rh.std()),
        k_mean=float(k.mean()), k_std=float(k.std()),
        k_min=int(k.min()), k_max=int(k.max())
    )


@torch.no_grad()
def test_with_rows(model, loader, device, amp, csv_path):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = c1 = c5 = 0
    loss_sum = 0.0
    rs=[]; rhs=[]; ks=[]; rows=[]
    for x,y,paths in loader:
        x,y=x.to(device),y.to(device)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=(amp and device.type=="cuda")):
            o=model(x); loss=ce(o.logits,y)
        a,b=topk_hits(o.logits,y)
        bs=y.size(0); total+=bs;c1+=a;c5+=b;loss_sum+=loss.item()*bs
        pred=o.logits.argmax(1).cpu().numpy()
        rr=model.last_r.detach().float().cpu().numpy()
        rh=model.last_actual_r.cpu().numpy()
        kk=model.last_k.cpu().numpy()
        yy=y.cpu().numpy()
        rs+=rr.tolist();rhs+=rh.tolist();ks+=kk.tolist()
        for i in range(bs):
            rows.append([paths[i],int(yy[i]),int(pred[i]),int(pred[i]==yy[i]),
                         float(rr[i]),float(rh[i]),int(kk[i])])
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["video","label","prediction","correct",
                                    "predicted_retention","actual_retention","k_per_frame"])
        w.writerows(rows)
    r=np.asarray(rs);rh=np.asarray(rhs);k=np.asarray(ks)
    return dict(loss=loss_sum/total,top1=100*c1/total,top5=100*c5/total,
                r_mean=float(r.mean()),r_std=float(r.std()),r_min=float(r.min()),
                r_max=float(r.max()),actual_r_mean=float(rh.mean()),
                actual_r_std=float(rh.std()),k_mean=float(k.mean()),
                k_std=float(k.std()),k_min=int(k.min()),k_max=int(k.max()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",required=True)
    ap.add_argument("--split-root",required=True)
    ap.add_argument("--dense-checkpoint",required=True)
    ap.add_argument("--e2-checkpoint",required=True)
    ap.add_argument("--split",type=int,default=1)
    ap.add_argument("--num-frames",type=int,default=8)
    ap.add_argument("--insert-block",type=int,default=6)
    ap.add_argument("--train-from-block",type=int,default=6)
    ap.add_argument("--r-min",type=float,default=.25)
    ap.add_argument("--r-max",type=float,default=.75)
    ap.add_argument("--target-rho",type=float,default=.50)
    ap.add_argument("--lambda-budget",type=float,default=5.0)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--batch-size",type=int,default=1)
    ap.add_argument("--accum-steps",type=int,default=4)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--lr-scorer",type=float,default=5e-5)
    ap.add_argument("--lr-budget",type=float,default=1e-4)
    ap.add_argument("--lr-backbone",type=float,default=5e-6)
    ap.add_argument("--weight-decay",type=float,default=.05)
    ap.add_argument("--val-fraction",type=float,default=.10)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--amp",action="store_true")
    ap.add_argument("--output",default="runs/e3_adaptive_r050")
    a=ap.parse_args()

    if not (0 < a.r_min < a.r_max <= 1 and a.r_min <= a.target_rho <= a.r_max):
        raise ValueError("Invalid retention range/target.")

    seed_all(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*88)
    print("AdaST E3 — CONTENT-ADAPTIVE TOKEN BUDGET")
    print("="*88)
    print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))
    print(f"Retention range: [{a.r_min:.2f},{a.r_max:.2f}], target={a.target_rho:.2f}")

    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    otr,te=read_split(a.data_root,a.split_root,a.split)
    tr,va=stratified_split(otr,a.val_fraction,a.seed)
    print(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    kw=dict(batch_size=a.batch_size,num_workers=a.workers,
            pin_memory=device.type=="cuda",persistent_workers=a.workers>0)
    trl=DataLoader(UCF(tr,proc,a.num_frames),shuffle=True,**kw)
    val=DataLoader(UCF(va,proc,a.num_frames),shuffle=False,**kw)
    tel=DataLoader(UCF(te,proc,a.num_frames),shuffle=False,**kw)

    bb=TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME,num_labels=101,ignore_mismatched_sizes=True)
    load_dense(bb,a.dense_checkpoint)
    model=AdaSTE3(bb,a.insert_block,a.num_frames,a.r_min,a.r_max)
    load_e2(model,a.e2_checkpoint)
    configure(model,a.train_from_block)
    model.to(device)

    scorer=list(model.scorer.parameters())
    budget=list(model.budget.parameters())
    other=[p for n,p in model.named_parameters() if p.requires_grad
           and not n.startswith("scorer.") and not n.startswith("budget.")]
    opt=torch.optim.AdamW([
        {"params":scorer,"lr":a.lr_scorer},
        {"params":budget,"lr":a.lr_budget},
        {"params":other,"lr":a.lr_backbone}],weight_decay=a.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    scaler=torch.amp.GradScaler("cuda") if a.amp and device.type=="cuda" else None

    met=out/"metrics.csv"
    with met.open("w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch","train_top1","val_top1","val_top5",
                                "val_r_mean","val_r_std","val_actual_r","val_k_mean"])

    best=-1e9;bestp=out/"best.pt"
    for ep in range(1,a.epochs+1):
        print(f"\nEPOCH {ep}/{a.epochs}")
        t0=time.time()
        trm=run_epoch(model,trl,device,a.amp,opt,scaler,a.accum_steps,
                      a.target_rho,a.lambda_budget)
        vam=run_epoch(model,val,device,a.amp)
        score=vam["top1"]-10*abs(vam["actual_r_mean"]-a.target_rho)
        print(f"Train Top-1={trm['top1']:.2f}%")
        print(f"Val Top-1={vam['top1']:.2f}% | Top-5={vam['top5']:.2f}%")
        print(f"rho mean/std={vam['r_mean']:.4f}/{vam['r_std']:.4f} | "
              f"actual={vam['actual_r_mean']:.4f} | K={vam['k_mean']:.2f}")
        print(f"Epoch time={time.time()-t0:.1f}s")
        with met.open("a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([ep,trm["top1"],vam["top1"],vam["top5"],
                                    vam["r_mean"],vam["r_std"],
                                    vam["actual_r_mean"],vam["k_mean"]])
        if score>best:
            best=score
            torch.save({"model_state_dict":model.state_dict(),"epoch":ep,
                        "val":vam,"args":vars(a)},bestp)
            print("Saved new best E3 checkpoint.")
        sched.step()

    ck=torch.load(bestp,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model_state_dict"],strict=True);model.to(device)
    tem=test_with_rows(model,tel,device,a.amp,out/"per_video_test.csv")

    summary={
        "experiment":"AdaST E3 - Content-Adaptive Per-Video Token Budget",
        "dataset":"UCF101 Split-1","backbone":"TimeSformer-Base",
        "dense_reference_top1":92.55,"dense_reference_top5":98.97,
        "e2_v2_reference_top1":89.92862807295796,
        "e2_v2_reference_top5":97.80597409463388,
        "r_min":a.r_min,"r_max":a.r_max,"target_rho":a.target_rho,
        "predicted_r_mean":tem["r_mean"],"predicted_r_std":tem["r_std"],
        "predicted_r_min":tem["r_min"],"predicted_r_max":tem["r_max"],
        "actual_retention_mean":tem["actual_r_mean"],
        "actual_retention_std":tem["actual_r_std"],
        "k_per_frame_mean":tem["k_mean"],"k_per_frame_std":tem["k_std"],
        "k_per_frame_min":tem["k_min"],"k_per_frame_max":tem["k_max"],
        "test_loss":tem["loss"],"test_top1":tem["top1"],"test_top5":tem["top5"],
        "physical_sequence_reduction":False,
        "note":"Shape-preserving adaptive hard gating; no physical pruning speedup claim.",
        "checkpoint":str(bestp.resolve()),"config":vars(a)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["top1","top5","target_rho","actual_r_mean",
                                    "actual_r_std","pred_r_mean","pred_r_std",
                                    "pred_r_min","pred_r_max","k_mean","k_std","k_min","k_max"])
        w.writerow([tem["top1"],tem["top5"],a.target_rho,tem["actual_r_mean"],
                    tem["actual_r_std"],tem["r_mean"],tem["r_std"],tem["r_min"],
                    tem["r_max"],tem["k_mean"],tem["k_std"],tem["k_min"],tem["k_max"]])

    print("\n"+"="*88)
    print("E3 ADAPTIVE BUDGET — FINAL TEST RESULT")
    print("="*88)
    print(f"Test Top-1            : {tem['top1']:.4f}%")
    print(f"Test Top-5            : {tem['top5']:.4f}%")
    print(f"Pred retention mean   : {tem['r_mean']:.4f}")
    print(f"Pred retention std    : {tem['r_std']:.4f}")
    print(f"Pred retention range  : [{tem['r_min']:.4f},{tem['r_max']:.4f}]")
    print(f"Actual retention mean : {tem['actual_r_mean']:.4f}")
    print(f"K/frame mean/std      : {tem['k_mean']:.2f}/{tem['k_std']:.2f}")
    print(f"K/frame range         : [{tem['k_min']},{tem['k_max']}]")
    print("Physical pruning      : NO (E3 ablation uses shape-preserving gating)")
    print("Summary:",(out/"summary.json").resolve())
    print("="*88)


if __name__=="__main__":
    main()
