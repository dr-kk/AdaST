#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaST E4 — Bounded Discrete Content-Adaptive Token Budget
TimeSformer / UCF101 Split-1

Purpose
-------
E2-v2 uses learned per-frame importance with a fixed 50% retention.
E3-v1 collapsed to K=98 for every clip because a batch-size-1 budget penalty
effectively forced each sample toward rho=0.50.

E3-v2 corrects that failure:
  1. Learned controller chooses among discrete retention levels:
       {0.375, 0.4375, 0.50, 0.5625, 0.625}
  2. Hard forward allocation uses one level per video.
  3. Straight-through Gumbel-Softmax trains the controller.
  4. Mean-compute control uses a detached-history running EMA, not a per-video rho=0.50 penalty.
  5. Entropy/diversity regularization discourages single-level collapse.

This remains shape-preserving hard gating, NOT physical token deletion.
Do not claim its latency as physical pruning speedup.
"""

import argparse, csv, json, os, random, time
from collections import defaultdict, Counter
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification

MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
LEVELS = (0.375, 0.4375, 0.50, 0.5625, 0.625)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_class_index(root):
    p = Path(root) / "classInd.txt"
    if not p.exists(): raise FileNotFoundError(f"Missing: {p}")
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            i, c = line.split(maxsplit=1); d[c] = int(i)-1
    return d


def read_split(data_root, split_root, split=1):
    dr, sr = Path(data_root), Path(split_root)
    cmap = read_class_index(sr)
    tf, ef = sr/f"trainlist{split:02d}.txt", sr/f"testlist{split:02d}.txt"
    if not tf.exists(): raise FileNotFoundError(tf)
    if not ef.exists(): raise FileNotFoundError(ef)
    tr, te = [], []
    for line in tf.read_text(encoding="utf-8").splitlines():
        z=line.strip().split()
        if z:
            rel=z[0].replace("/",os.sep)
            tr.append((dr/rel,cmap[Path(rel).parts[0]]))
    for line in ef.read_text(encoding="utf-8").splitlines():
        rel=line.strip().replace("/",os.sep)
        if rel: te.append((dr/rel,cmap[Path(rel).parts[0]]))
    return tr,te


def stratified(items, frac=.1, seed=42):
    g=defaultdict(list)
    for x in items:g[x[1]].append(x)
    rng=random.Random(seed); tr=[];va=[]
    for y in sorted(g):
        a=g[y][:];rng.shuffle(a)
        nv=max(1,min(len(a)-1,round(len(a)*frac)))
        va+=a[:nv];tr+=a[nv:]
    rng.shuffle(tr);rng.shuffle(va)
    return tr,va


def decode(path,T):
    try:c=av.open(str(path))
    except Exception as e:raise RuntimeError(f"Could not open {path}\n{e}") from e
    xs=[]
    try:
        s=c.streams.video[0];s.thread_type="AUTO"
        for f in c.decode(s):xs.append(f.to_ndarray(format="rgb24"))
    finally:c.close()
    if not xs:raise RuntimeError(f"No frames: {path}")
    ids=np.linspace(0,len(xs)-1,T).round().astype(int)
    return [xs[i] for i in ids]


class UCF(Dataset):
    def __init__(self,items,processor,T):
        self.items,self.processor,self.T=items,processor,T
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]
        x=self.processor(images=decode(p,self.T),return_tensors="pt")["pixel_values"].squeeze(0)
        return x,int(y),str(p)


def hits(logits,y):
    _,p=logits.topk(min(5,logits.size(1)),1)
    return p[:,:1].eq(y[:,None]).sum().item(),p.eq(y[:,None]).any(1).sum().item()


class ImportanceScorer(nn.Module):
    def __init__(self,d=768,h=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3*d,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,1))
    def forward(self,patch,T):
        b,m,d=patch.shape;n=m//T;z=patch.reshape(b,T,n,d)
        delta=torch.zeros_like(z);delta[:,1:]=z[:,1:]-z[:,:-1]
        ctx=z.mean((1,2),keepdim=True).expand(-1,T,n,-1)
        s=torch.sigmoid(self.net(torch.cat([z,delta,ctx],-1))).squeeze(-1)
        return s.reshape(b,m),delta


class DiscreteBudgetNet(nn.Module):
    def __init__(self,d=768,h=128,nlevels=5):
        super().__init__()
        self.cls=nn.Sequential(nn.Linear(d,64),nn.GELU())
        self.net=nn.Sequential(nn.Linear(67,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,nlevels))
        nn.init.normal_(self.net[-1].weight,0.0,1e-3)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,scores,delta,cls):
        sm=scores.mean(1,keepdim=True)
        ss=scores.std(1,keepdim=True,unbiased=False)
        tm=delta.abs().mean((1,2,3)).unsqueeze(1)
        return self.net(torch.cat([sm,ss,tm,self.cls(cls.squeeze(1))],1))


class TemporalBudgetNet(nn.Module):
    """Content-adaptive temporal budget controller over S in {4,5,6,7,8}."""
    def __init__(self, d=768, h=128, nlevels=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*d, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, nlevels)
        )
        nn.init.normal_(self.net[-1].weight, 0.0, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        # z: [B,T,N,D]. Combine global appearance and temporal-change descriptors.
        appearance = z.mean((1,2))
        frame = z.mean(2)
        motion = torch.zeros_like(frame)
        motion[:,1:] = frame[:,1:] - frame[:,:-1]
        temporal = motion.abs().mean(1)
        return self.net(torch.cat([appearance, temporal], dim=1))


class AdaSTE4(nn.Module):
    def __init__(self,backbone,block=6,T=8,temperature=1.0):
        super().__init__()
        self.backbone=backbone;self.block=block;self.T=T;self.temperature=temperature
        d=backbone.config.hidden_size
        self.scorer=ImportanceScorer(d,256)
        self.budget=DiscreteBudgetNet(d,128,len(LEVELS))
        self.temporal_budget=TemporalBudgetNet(d,128,5)
        self.register_buffer('temporal_levels', torch.tensor([4,5,6,7,8], dtype=torch.float32))
        self.register_buffer("levels",torch.tensor(LEVELS,dtype=torch.float32))
        self.last_probs=self.last_r=self.last_actual_r=self.last_k=self.last_idx=None
        self.last_t_probs=self.last_s=self.last_t_idx=None
        self.layers()[block].register_forward_pre_hook(self._hook,with_kwargs=True)

    def layers(self):return self.backbone.timesformer.encoder.layer

    def _hook(self,module,args,kwargs):
        if args:h,rest,src=args[0],args[1:],"args"
        else:h,rest,src=kwargs["hidden_states"],(),"kwargs"
        cls,patch=h[:,:1],h[:,1:];b,m,d=patch.shape
        if m%self.T:raise RuntimeError(f"{m} patch tokens not divisible by T={self.T}")
        n=m//self.T
        # E4 temporal controller.
        z = patch.reshape(b,self.T,n,d)
        t_logits = self.temporal_budget(z)
        if self.training:
            t_choice = F.gumbel_softmax(t_logits, tau=self.temperature, hard=True, dim=1)
        else:
            t_idx_eval = t_logits.argmax(1)
            t_choice = F.one_hot(t_idx_eval, num_classes=5).to(t_logits.dtype)
        t_probs = torch.softmax(t_logits, dim=1)
        s_budget = (t_choice * self.temporal_levels[None,:]).sum(1)
        t_idx = t_choice.argmax(1)

        # Rank frames by temporal novelty; preserve exactly S(X) frames.
        frame_feat = z.mean(2)
        novelty = torch.zeros((b,self.T), device=z.device, dtype=z.dtype)
        novelty[:,0] = frame_feat[:,0].pow(2).mean(-1).sqrt()
        if self.T > 1:
            novelty[:,1:] = (frame_feat[:,1:] - frame_feat[:,:-1]).pow(2).mean(-1).sqrt()
        frame_mask = torch.zeros_like(novelty)
        for bi in range(b):
            si = int(s_budget[bi].item())
            ids = torch.topk(novelty[bi], si, largest=True, sorted=False).indices
            frame_mask[bi].scatter_(0, ids, 1.0)

        # Straight-through temporal gate; forward is hard, backward follows expected S.
        expected_s = (t_probs * self.temporal_levels[None,:]).sum(1)
        t_st = 1.0 + expected_s / float(self.T) - (expected_s / float(self.T)).detach()
        z = z * frame_mask[:,:,None,None] * t_st[:,None,None,None]
        patch = z.reshape(b,m,d)

        scores,delta=self.scorer(patch,self.T)
        logits=self.budget(scores,delta,cls)

        if self.training:
            choice=F.gumbel_softmax(logits,tau=self.temperature,hard=True,dim=1)
        else:
            idx=logits.argmax(1)
            choice=F.one_hot(idx,num_classes=len(LEVELS)).to(logits.dtype)

        probs=torch.softmax(logits,dim=1)
        r=(choice*self.levels[None,:]).sum(1)
        idx=choice.argmax(1)
        k=torch.round(r*n).long().clamp(1,n)

        sf=scores.reshape(b,self.T,n);mf=torch.zeros_like(sf)
        for bi in range(b):
            ids=torch.topk(sf[bi],int(k[bi]),dim=1,largest=True,sorted=False).indices
            mf[bi].scatter_(1,ids,1.0)
        hard=mf.reshape(b,m)

        score_gate=hard+scores-scores.detach()
        # Controller gradient surrogate without altering the hard forward gate.
        expected_r=(probs*self.levels[None,:]).sum(1)
        budget_st=1.0+expected_r-expected_r.detach()
        gate=score_gate*budget_st[:,None]
        gh=torch.cat([cls,patch*gate.unsqueeze(-1)],1)

        self.last_probs=probs
        self.last_r=r.detach()
        self.last_expected_r=expected_r
        self.last_actual_r=hard.mean(1).detach()
        self.last_k=k.detach();self.last_idx=idx.detach()
        self.last_t_probs=t_probs
        self.last_s=s_budget.detach()
        self.last_expected_s=expected_s
        self.last_t_idx=t_idx.detach()

        if src=="args":return (gh,)+rest,kwargs
        kw=dict(kwargs);kw["hidden_states"]=gh;return args,kw

    def forward(self,x):return self.backbone(pixel_values=x,return_dict=True)


def state(obj):
    if isinstance(obj,dict):
        for k in ("model_state_dict","state_dict","model"):
            if k in obj and isinstance(obj[k],dict):return obj[k]
        return obj
    raise RuntimeError("Unsupported checkpoint")


def norm(sd):
    out={}
    for k,v in sd.items():
        while k.startswith("module."):k=k[7:]
        out[k]=v
    return out


def load_dense(bb,path):
    sd=norm(state(torch.load(path,map_location="cpu",weights_only=False)))
    if any(k.startswith("backbone.") for k in sd):
        sd={k[9:] if k.startswith("backbone.") else k:v for k,v in sd.items()}
    z=bb.load_state_dict(sd,strict=False)
    print(f"Dense checkpoint: missing={len(z.missing_keys)}, unexpected={len(z.unexpected_keys)}")


def load_e3(model,path):
    sd=norm(state(torch.load(path,map_location="cpu",weights_only=False)))
    # E3-v3 checkpoint contains backbone + scorer + spatial budget.
    # Only the new E4 temporal controller is allowed to be missing.
    z=model.load_state_dict(sd,strict=False)
    bad_missing=[k for k in z.missing_keys if not k.startswith("temporal_budget.") and k!="temporal_levels"]
    if bad_missing:
        raise RuntimeError("Unexpected missing E3-v3 keys: "+str(bad_missing[:20]))
    if z.unexpected_keys:
        raise RuntimeError("Unexpected E3-v3 keys: "+str(z.unexpected_keys[:20]))
    print(f"E3-v3 -> E4-v2: missing={len(z.missing_keys)}, unexpected={len(z.unexpected_keys)}")
    print("Spatial scorer/controller loaded from E3-v3 and will remain frozen.")


def configure(model,start=10):
    # Freeze the successful E3-v3 spatial pathway completely.
    for p in model.parameters(): p.requires_grad=False
    for p in model.temporal_budget.parameters(): p.requires_grad=True
    # Limited downstream adaptation only; default blocks 10-11 + classifier/layernorm.
    for i in range(start,len(model.layers())):
        for p in model.layers()[i].parameters(): p.requires_grad=True
    for p in model.backbone.classifier.parameters(): p.requires_grad=True
    for p in model.backbone.timesformer.layernorm.parameters(): p.requires_grad=True


def epoch(model,loader,device,amp,opt=None,scaler=None,accum=4,target_s=6.5,
          lam_s=0.20,lam_div=0.05,ema_decay=.99):
    training=opt is not None;model.train(training);ce=nn.CrossEntropyLoss()
    if training:opt.zero_grad(set_to_none=True)
    total=c1=c5=0;ls=0.;rs=[];rhs=[];ks=[];idxs=[];ss=[];tidxs=[]
    ema_s=torch.tensor(target_s,device=device)
    context=torch.enable_grad if training else torch.no_grad
    with context():
        for step,(x,y,_) in enumerate(loader,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
                o=model(x);cl=ce(o.logits,y)
                if training:
                    es=model.last_expected_s.mean()
                    ema_s=ema_s.detach()
                    ema_s=ema_decay*ema_s+(1-ema_decay)*es
                    s_loss=(ema_s-target_s).pow(2)
                    tp=model.last_t_probs.mean(0)
                    t_div=(tp*torch.log(tp.clamp_min(1e-8))).sum()
                    loss=cl+lam_s*s_loss+lam_div*t_div
                else: loss=cl
            if training:
                q=loss/accum
                if scaler:scaler.scale(q).backward()
                else:q.backward()
                if step%accum==0 or step==len(loader):
                    pars=[p for p in model.parameters() if p.requires_grad]
                    if scaler:
                        scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(pars,1.)
                        scaler.step(opt);scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(pars,1.);opt.step()
                    opt.zero_grad(set_to_none=True)
            a,b=hits(o.logits.detach(),y);bs=y.size(0)
            total+=bs;c1+=a;c5+=b;ls+=float(loss.item())*bs
            rs+=model.last_r.float().cpu().tolist();rhs+=model.last_actual_r.float().cpu().tolist()
            ks+=model.last_k.cpu().tolist();idxs+=model.last_idx.cpu().tolist()
            ss+=model.last_s.cpu().tolist();tidxs+=model.last_t_idx.cpu().tolist()
            if training and (step%200==0 or step==len(loader)):
                ar=np.asarray(rs);az=np.asarray(ss)
                print(f"  batch {step}/{len(loader)} | loss {ls/total:.4f} | top1 {100*c1/total:.2f}% "
                      f"| K-rho {ar.mean():.4f}/{ar.std():.4f} | S {az.mean():.3f}/{az.std():.3f} "
                      f"| S-counts {dict(Counter(ss))}")
    r=np.asarray(rs);rh=np.asarray(rhs);k=np.asarray(ks);sv=np.asarray(ss);cnt=Counter(idxs);tcnt=Counter(tidxs)
    combined=r*(sv/float(model.T))
    return dict(loss=ls/total,top1=100*c1/total,top5=100*c5/total,
                r_mean=float(r.mean()),r_std=float(r.std()),r_min=float(r.min()),r_max=float(r.max()),
                actual_r_mean=float(rh.mean()),k_mean=float(k.mean()),k_std=float(k.std()),k_min=int(k.min()),k_max=int(k.max()),
                level_counts={str(LEVELS[i]):int(cnt.get(i,0)) for i in range(len(LEVELS))},
                s_mean=float(sv.mean()),s_std=float(sv.std()),s_min=int(sv.min()),s_max=int(sv.max()),
                temporal_level_counts={str(i+4):int(tcnt.get(i,0)) for i in range(5)},
                combined_retention_mean=float(combined.mean()),combined_retention_std=float(combined.std()))


@torch.no_grad()
def final_test(model,loader,device,amp,path):
    model.eval();ce=nn.CrossEntropyLoss();total=c1=c5=0;ls=0
    rs=[];rhs=[];ks=[];idxs=[];ss=[];tidxs=[];rows=[]
    for x,y,paths in loader:
        x=x.to(device);y=y.to(device)
        with torch.autocast("cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
            o=model(x);loss=ce(o.logits,y)
        a,b=hits(o.logits,y);bs=y.size(0);total+=bs;c1+=a;c5+=b;ls+=loss.item()*bs
        pred=o.logits.argmax(1).cpu().numpy();yy=y.cpu().numpy()
        rr=model.last_r.cpu().numpy();rh=model.last_actual_r.cpu().numpy();kk=model.last_k.cpu().numpy();ii=model.last_idx.cpu().numpy()
        sv=model.last_s.cpu().numpy();ti=model.last_t_idx.cpu().numpy()
        rs+=rr.tolist();rhs+=rh.tolist();ks+=kk.tolist();idxs+=ii.tolist();ss+=sv.tolist();tidxs+=ti.tolist()
        for j in range(bs):
            comb=float(rr[j])*(float(sv[j])/float(model.T))
            rows.append([paths[j],int(yy[j]),int(pred[j]),int(pred[j]==yy[j]),float(rr[j]),int(kk[j]),int(sv[j]),comb])
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["video","label","prediction","correct","spatial_retention","k_per_frame","temporal_S","combined_effective_retention"]);w.writerows(rows)
    r=np.asarray(rs);rh=np.asarray(rhs);k=np.asarray(ks);sv=np.asarray(ss);cnt=Counter(idxs);tcnt=Counter(tidxs);combined=r*(sv/float(model.T))
    return dict(loss=ls/total,top1=100*c1/total,top5=100*c5/total,r_mean=float(r.mean()),r_std=float(r.std()),r_min=float(r.min()),r_max=float(r.max()),
                actual_r_mean=float(rh.mean()),k_mean=float(k.mean()),k_std=float(k.std()),k_min=int(k.min()),k_max=int(k.max()),
                level_counts={str(LEVELS[i]):int(cnt.get(i,0)) for i in range(len(LEVELS))},
                s_mean=float(sv.mean()),s_std=float(sv.std()),s_min=int(sv.min()),s_max=int(sv.max()),
                temporal_level_counts={str(i+4):int(tcnt.get(i,0)) for i in range(5)},
                combined_retention_mean=float(combined.mean()),combined_retention_std=float(combined.std()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True);p.add_argument("--split-root",required=True)
    p.add_argument("--dense-checkpoint",required=True);p.add_argument("--e3-checkpoint",required=True)
    p.add_argument("--split",type=int,default=1);p.add_argument("--num-frames",type=int,default=8)
    p.add_argument("--insert-block",type=int,default=6);p.add_argument("--train-from-block",type=int,default=10)
    p.add_argument("--target-rho",type=float,default=.50);p.add_argument("--target-s",type=float,default=6.5);p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--lambda-s",type=float,default=.20);p.add_argument("--lambda-div",type=float,default=.05)
    p.add_argument("--ema-decay",type=float,default=.99);p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--accum-steps",type=int,default=4)
    p.add_argument("--workers",type=int,default=2);p.add_argument("--lr-scorer",type=float,default=5e-5)
    p.add_argument("--lr-budget",type=float,default=1e-4);p.add_argument("--lr-backbone",type=float,default=5e-6)
    p.add_argument("--weight-decay",type=float,default=.05);p.add_argument("--val-fraction",type=float,default=.10)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--amp",action="store_true")
    p.add_argument("--output",default="runs/e4_v2_temporal_frozen_spatial");a=p.parse_args()

    seed_all(a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*88);print("AdaST E4-v2 — FROZEN SPATIAL + ADAPTIVE TEMPORAL BUDGET");print("="*88)
    print("Levels:",LEVELS);print("Target mean:",a.target_rho);print("Device:",device)
    if device.type=="cuda":print("GPU:",torch.cuda.get_device_name(0))

    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    otr,te=read_split(a.data_root,a.split_root,a.split);tr,va=stratified(otr,a.val_fraction,a.seed)
    print(f"Train={len(tr)}, Val={len(va)}, Test={len(te)}")
    kw=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=="cuda",
            persistent_workers=a.workers>0)
    trl=DataLoader(UCF(tr,proc,a.num_frames),shuffle=True,**kw)
    val=DataLoader(UCF(va,proc,a.num_frames),shuffle=False,**kw)
    tel=DataLoader(UCF(te,proc,a.num_frames),shuffle=False,**kw)

    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_labels=101,ignore_mismatched_sizes=True)
    load_dense(bb,a.dense_checkpoint)
    model=AdaSTE4(bb,a.insert_block,a.num_frames,a.temperature)
    load_e3(model,a.e3_checkpoint);configure(model,a.train_from_block);model.to(device)

    tp=list(model.temporal_budget.parameters())
    op=[q for n,q in model.named_parameters() if q.requires_grad and not n.startswith("temporal_budget.")]
    opt=torch.optim.AdamW([{"params":tp,"lr":a.lr_budget},{"params":op,"lr":a.lr_backbone}],weight_decay=a.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    scaler=torch.amp.GradScaler("cuda") if a.amp and device.type=="cuda" else None

    best=-1e9;bestp=out/"best.pt";mp=out/"metrics.csv"
    with mp.open("w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch","train_top1","val_top1","val_top5","val_r_mean","val_r_std","val_k_mean","val_k_std","val_s_mean","val_s_std","combined_retention","spatial_counts","temporal_counts"])
    for ep in range(1,a.epochs+1):
        print(f"\nEPOCH {ep}/{a.epochs}");t=time.time()
        trm=epoch(model,trl,device,a.amp,opt,scaler,a.accum_steps,a.target_s,a.lambda_s,a.lambda_div,a.ema_decay)
        vam=epoch(model,val,device,a.amp,target_s=a.target_s)
        print(f"Train Top-1={trm['top1']:.2f}%")
        print(f"Val Top-1={vam['top1']:.2f}% | Top-5={vam['top5']:.2f}%")
        print(f"rho={vam['r_mean']:.4f} +/- {vam['r_std']:.4f} | K={vam['k_mean']:.2f} +/- {vam['k_std']:.2f}")
        print("Spatial counts:",vam["level_counts"]);print(f"Temporal S={vam['s_mean']:.3f} +/- {vam['s_std']:.3f} range=[{vam['s_min']},{vam['s_max']}]");print("Temporal counts:",vam["temporal_level_counts"]);print(f"Combined effective retention={vam['combined_retention_mean']:.4f}");print(f"Epoch time={time.time()-t:.1f}s")
        with mp.open("a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([ep,trm["top1"],vam["top1"],vam["top5"],vam["r_mean"],vam["r_std"],vam["k_mean"],vam["k_std"],vam["s_mean"],vam["s_std"],vam["combined_retention_mean"],json.dumps(vam["level_counts"]),json.dumps(vam["temporal_level_counts"])])
        # Require accuracy and reward genuine adaptation while controlling mean compute.
        score=vam["top1"]-0.5*abs(vam["s_mean"]-a.target_s)+0.10*min(vam["s_std"],1.0)
        if score>best:
            best=score;torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"val":vam,"args":vars(a)},bestp)
            print("Saved new best E4-v2 checkpoint.")
        sch.step()

    ck=torch.load(bestp,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model_state_dict"],strict=True);model.to(device)
    te_m=final_test(model,tel,device,a.amp,out/"per_video_test.csv")
    summary={"experiment":"AdaST E4-v2 - Frozen E3-v3 Spatial + Adaptive Temporal Budget","dataset":"UCF101 Split-1",
             "spatial_levels":LEVELS,"temporal_levels":[4,5,6,7,8],"target_s":a.target_s,
             "e3_v3_reference_top1":90.1665,"test_top1":te_m["top1"],"test_top5":te_m["top5"],
             "spatial_retention_mean":te_m["r_mean"],"spatial_retention_std":te_m["r_std"],
             "k_mean":te_m["k_mean"],"k_std":te_m["k_std"],"k_min":te_m["k_min"],"k_max":te_m["k_max"],
             "spatial_level_counts":te_m["level_counts"],"temporal_s_mean":te_m["s_mean"],"temporal_s_std":te_m["s_std"],
             "temporal_s_min":te_m["s_min"],"temporal_s_max":te_m["s_max"],"temporal_level_counts":te_m["temporal_level_counts"],
             "combined_effective_retention_mean":te_m["combined_retention_mean"],
             "combined_effective_retention_std":te_m["combined_retention_std"],
             "physical_sequence_reduction":False,"checkpoint":str(bestp.resolve()),"config":vars(a)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["top1","top5","spatial_r_mean","spatial_r_std","k_mean","k_std","k_min","k_max","s_mean","s_std","s_min","s_max","combined_retention","spatial_counts","temporal_counts"])
        w.writerow([te_m["top1"],te_m["top5"],te_m["r_mean"],te_m["r_std"],te_m["k_mean"],te_m["k_std"],te_m["k_min"],te_m["k_max"],te_m["s_mean"],te_m["s_std"],te_m["s_min"],te_m["s_max"],te_m["combined_retention_mean"],json.dumps(te_m["level_counts"]),json.dumps(te_m["temporal_level_counts"])])


    print("\n"+"="*88);print("E4-v2 FROZEN-SPATIAL + ADAPTIVE-TEMPORAL — FINAL TEST RESULT");print("="*88)
    print(f"Test Top-1       : {te_m['top1']:.4f}%");print(f"Test Top-5       : {te_m['top5']:.4f}%")
    print(f"Retention mean   : {te_m['r_mean']:.4f}");print(f"Retention std    : {te_m['r_std']:.4f}")
    print(f"Retention range  : [{te_m['r_min']:.4f}, {te_m['r_max']:.4f}]")
    print(f"K/frame mean/std : {te_m['k_mean']:.2f}/{te_m['k_std']:.2f}")
    print(f"K/frame range    : [{te_m['k_min']},{te_m['k_max']}]")
    print("Spatial counts   :",te_m["level_counts"]);print(f"Temporal S mean/std: {te_m['s_mean']:.3f}/{te_m['s_std']:.3f}");print(f"Temporal S range : [{te_m['s_min']},{te_m['s_max']}]");print("Temporal counts  :",te_m["temporal_level_counts"]);print(f"Combined retention: {te_m['combined_retention_mean']:.4f}")
    print("Physical pruning : NO — shape-preserving E4-v2 ablation")
    print("Summary          :", (out/"summary.json").resolve());print("="*88)


if __name__=="__main__":main()
