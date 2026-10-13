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
    """Content-adaptive temporal budget controller over S in {6,7,8}."""
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



class AdaptiveTokenConsolidator(nn.Module):
    """
    E5 content-adaptive soft token consolidation.

    Non-retained joint spatio-temporal tokens are not simply discarded:
    their information is softly assigned to retained representatives by
    learned feature similarity and aggregated as a residual.

    IMPORTANT: sequence shape is still preserved in this ablation. Therefore
    reported retention is an effective token budget, not measured physical
    pruning/FLOP reduction.
    """
    def __init__(self, d=768, proj_dim=96, temperature=0.20):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, proj_dim, bias=False)
        self.k = nn.Linear(d, proj_dim, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(2*d, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.temperature = float(temperature)

        # Start close to E5: small residual contribution, then learn.
        nn.init.normal_(self.q.weight, std=0.02)
        nn.init.normal_(self.k.weight, std=0.02)
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, pristine_patch, keep_mask):
        # pristine_patch [B,M,D], keep_mask [B,M] in {0,1}
        b,m,d = pristine_patch.shape
        out = pristine_patch * keep_mask.unsqueeze(-1)
        gate_values = []
        assignment_entropies = []

        # Batch size is normally 1 for GTX 1070 Ti; loop keeps memory bounded.
        for bi in range(b):
            keep_idx = torch.nonzero(keep_mask[bi] > 0.5, as_tuple=False).squeeze(1)
            drop_idx = torch.nonzero(keep_mask[bi] <= 0.5, as_tuple=False).squeeze(1)
            if keep_idx.numel() == 0 or drop_idx.numel() == 0:
                gate_values.append(torch.zeros((), device=pristine_patch.device, dtype=pristine_patch.dtype))
                assignment_entropies.append(torch.zeros((), device=pristine_patch.device, dtype=pristine_patch.dtype))
                continue

            retained = pristine_patch[bi, keep_idx]       # [R,D]
            dropped  = pristine_patch[bi, drop_idx]       # [Q,D]

            q = F.normalize(self.q(self.norm(dropped)), dim=-1)
            k = F.normalize(self.k(self.norm(retained)), dim=-1)
            sim = (q @ k.transpose(0,1)) / self.temperature
            assign = torch.softmax(sim, dim=1)            # [Q,R]

            # Normalize by received assignment mass so representation magnitude
            # does not grow with the number of discarded tokens.
            values = self.v(self.norm(dropped))
            mass = assign.sum(0).clamp_min(1e-6)          # [R]
            merged = assign.transpose(0,1) @ values       # [R,D]
            merged = merged / mass.unsqueeze(1)

            # Content-adaptive merge strength for every retained representative.
            g = torch.sigmoid(self.gate(torch.cat([retained, merged], dim=-1)))
            updated = retained + g * merged

            # Functional index_copy keeps autograd intact.
            out_b = out[bi].index_copy(0, keep_idx, updated)
            out = torch.cat([out[:bi], out_b.unsqueeze(0), out[bi+1:]], dim=0)

            gate_values.append(g.mean())
            ent = -(assign * torch.log(assign.clamp_min(1e-8))).sum(1).mean()
            assignment_entropies.append(ent)

        self.last_gate = torch.stack(gate_values)
        self.last_assignment_entropy = torch.stack(assignment_entropies)
        return out

class AdaSTE6Profile(nn.Module):
    def __init__(self,backbone,block=6,T=8,temperature=1.0):
        super().__init__()
        self.backbone=backbone;self.block=block;self.T=T;self.temperature=temperature
        d=backbone.config.hidden_size
        self.scorer=ImportanceScorer(d,256)
        self.budget=DiscreteBudgetNet(d,128,len(LEVELS))
        self.temporal_budget=TemporalBudgetNet(d,128,3)
        self.register_buffer('temporal_levels', torch.tensor([6,7,8], dtype=torch.float32))
        self.consolidator=AdaptiveTokenConsolidator(d=d,proj_dim=96,temperature=0.20)
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

        # ------------------------------------------------------------------
        # 1) FROZEN E3-v3 SPATIAL POLICY ON PRISTINE FEATURES.
        # Temporal masking must NOT change the inputs seen by K(X).
        # ------------------------------------------------------------------
        pristine_patch = patch
        pristine_z = pristine_patch.reshape(b,self.T,n,d)
        scores,delta=self.scorer(pristine_patch,self.T)
        logits=self.budget(scores,delta,cls)

        # Spatial controller is frozen; use deterministic E3-v3 decisions in
        # both train and evaluation so its policy cannot drift due to Gumbel noise.
        idx=logits.argmax(1)
        choice=F.one_hot(idx,num_classes=len(LEVELS)).to(logits.dtype)
        probs=torch.softmax(logits,dim=1)
        r=(choice*self.levels[None,:]).sum(1)
        k=torch.round(r*n).long().clamp(1,n)

        sf=scores.reshape(b,self.T,n)
        spatial_mask=torch.zeros_like(sf)
        for bi in range(b):
            ids=torch.topk(sf[bi],int(k[bi]),dim=1,largest=True,sorted=False).indices
            spatial_mask[bi].scatter_(1,ids,1.0)

        # Frozen spatial gate: no surrogate gradient is needed for scorer/budget.
        spatial_hard=spatial_mask.reshape(b,m)
        spatial_patch=pristine_patch*spatial_hard.unsqueeze(-1)

        # ------------------------------------------------------------------
        # 2) E5 TEMPORAL POLICY ON THE SAME PRISTINE FEATURES.
        # Conservative bounded choices S(X) in {6,7,8}.
        # ------------------------------------------------------------------
        t_logits=self.temporal_budget(pristine_z)
        t_probs=torch.softmax(t_logits,dim=1)
        if self.training:
            t_choice=F.gumbel_softmax(t_logits,tau=self.temperature,hard=True,dim=1)
        else:
            t_idx_eval=t_logits.argmax(1)
            t_choice=F.one_hot(
                t_idx_eval,num_classes=len(self.temporal_levels)
            ).to(t_logits.dtype)

        s_budget=(t_choice*self.temporal_levels[None,:]).sum(1)
        t_idx=t_choice.argmax(1)
        expected_s=(t_probs*self.temporal_levels[None,:]).sum(1)

        # Temporal novelty is measured from pristine frame features.
        frame_feat=pristine_z.mean(2)
        novelty=torch.zeros((b,self.T),device=pristine_z.device,dtype=pristine_z.dtype)
        novelty[:,0]=frame_feat[:,0].pow(2).mean(-1).sqrt()
        if self.T>1:
            novelty[:,1:]=(frame_feat[:,1:]-frame_feat[:,:-1]).pow(2).mean(-1).sqrt()

        frame_mask=torch.zeros_like(novelty)
        for bi in range(b):
            si=int(s_budget[bi].item())
            ids=torch.topk(novelty[bi],si,largest=True,sorted=False).indices
            frame_mask[bi].scatter_(0,ids,1.0)

        # Hard forward temporal selection with a straight-through controller
        # surrogate. This remains shape-preserving gating, not physical pruning.
        t_st=1.0+expected_s/float(self.T)-(expected_s/float(self.T)).detach()
        temporal_gate=frame_mask[:,:,None].expand(-1,-1,n).reshape(b,m)
        joint_keep=spatial_hard*temporal_gate

        # E5: consolidate information from non-retained tokens into the retained
        # representatives instead of replacing all discarded information by zero.
        # K(X) and S(X) decisions remain exactly the E4-v4 decisions.
        joint_patch=self.consolidator(pristine_patch,joint_keep)

        # Straight-through temporal-controller surrogate remains available during
        # training without changing the hard forward token budget.
        joint_patch=joint_patch*t_st[:,None,None]
        gh=torch.cat([cls,joint_patch],1)

        self.last_probs=probs
        self.last_r=r.detach()
        self.last_expected_r=(probs*self.levels[None,:]).sum(1)
        self.last_actual_r=spatial_hard.mean(1).detach()
        self.last_k=k.detach();self.last_idx=idx.detach()
        self.last_t_probs=t_probs
        self.last_s=s_budget.detach()
        self.last_expected_s=expected_s
        self.last_t_idx=t_idx.detach()
        self.last_merge_gate=self.consolidator.last_gate.detach()
        self.last_merge_entropy=self.consolidator.last_assignment_entropy.detach()

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


def load_e5(model,path):
    sd=norm(state(torch.load(path,map_location="cpu",weights_only=False)))
    z=model.load_state_dict(sd,strict=False)
    if z.missing_keys or z.unexpected_keys:
        raise RuntimeError(f"E5 checkpoint mismatch: missing={z.missing_keys[:20]}, unexpected={z.unexpected_keys[:20]}")
    print("Loaded validated E5 checkpoint exactly.")


def configure(model,start=10):
    # Isolate M(X): freeze validated K(X) and S(X) controllers and upstream model.
    for p in model.parameters(): p.requires_grad=False
    for p in model.consolidator.parameters(): p.requires_grad=True
    # Limited downstream adaptation for consuming consolidated representations.
    for i in range(start,len(model.layers())):
        for p in model.layers()[i].parameters(): p.requires_grad=True
    for p in model.backbone.classifier.parameters(): p.requires_grad=True
    for p in model.backbone.timesformer.layernorm.parameters(): p.requires_grad=True


def epoch(model,loader,device,amp,opt=None,scaler=None,accum=4,target_s=7.0,
          lam_s=0.20,lam_div=0.15,ema_decay=.99):
    training=opt is not None;model.train(training);ce=nn.CrossEntropyLoss()
    if training:opt.zero_grad(set_to_none=True)
    total=c1=c5=0;ls=0.;rs=[];rhs=[];ks=[];idxs=[];ss=[];tidxs=[];mgs=[];mes=[]
    ema_s=torch.tensor(target_s,device=device)
    context=torch.enable_grad if training else torch.no_grad
    with context():
        for step,(x,y,_) in enumerate(loader,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
                o=model(x);cl=ce(o.logits,y)
                if training:
                    # E5 isolates consolidation M(X). K(X) and S(X) are frozen;
                    # optimize classification loss only.
                    loss=cl
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
            mgs+=model.last_merge_gate.cpu().tolist()
            mes+=model.last_merge_entropy.cpu().tolist()
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
                temporal_level_counts={str(level):int(tcnt.get(i,0)) for i,level in enumerate((6,7,8))},
                combined_retention_mean=float(combined.mean()),combined_retention_std=float(combined.std()),
                merge_gate_mean=float(np.mean(mgs)),merge_gate_std=float(np.std(mgs)),
                merge_assignment_entropy_mean=float(np.mean(mes)))


@torch.no_grad()
def final_test(model,loader,device,amp,path):
    model.eval();ce=nn.CrossEntropyLoss();total=c1=c5=0;ls=0
    rs=[];rhs=[];ks=[];idxs=[];ss=[];tidxs=[];mgs=[];mes=[];rows=[]
    for x,y,paths in loader:
        x=x.to(device);y=y.to(device)
        with torch.autocast("cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
            o=model(x);loss=ce(o.logits,y)
        a,b=hits(o.logits,y);bs=y.size(0);total+=bs;c1+=a;c5+=b;ls+=loss.item()*bs
        pred=o.logits.argmax(1).cpu().numpy();yy=y.cpu().numpy()
        rr=model.last_r.cpu().numpy();rh=model.last_actual_r.cpu().numpy();kk=model.last_k.cpu().numpy();ii=model.last_idx.cpu().numpy()
        sv=model.last_s.cpu().numpy();ti=model.last_t_idx.cpu().numpy()
        rs+=rr.tolist();rhs+=rh.tolist();ks+=kk.tolist();idxs+=ii.tolist();ss+=sv.tolist();tidxs+=ti.tolist();mgs+=model.last_merge_gate.cpu().tolist();mes+=model.last_merge_entropy.cpu().tolist()
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
                temporal_level_counts={str(level):int(tcnt.get(i,0)) for i,level in enumerate((6,7,8))},
                combined_retention_mean=float(combined.mean()),combined_retention_std=float(combined.std()),merge_gate_mean=float(np.mean(mgs)),merge_gate_std=float(np.std(mgs)),merge_assignment_entropy_mean=float(np.mean(mes)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True);p.add_argument("--split-root",required=True)
    p.add_argument("--dense-checkpoint",required=True);p.add_argument("--e5-checkpoint",required=True)
    p.add_argument("--split",type=int,default=1);p.add_argument("--num-frames",type=int,default=8)
    p.add_argument("--insert-block",type=int,default=6);p.add_argument("--train-from-block",type=int,default=10)
    p.add_argument("--target-rho",type=float,default=.50);p.add_argument("--target-s",type=float,default=7.0);p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--lambda-s",type=float,default=.20);p.add_argument("--lambda-div",type=float,default=.15)
    p.add_argument("--ema-decay",type=float,default=.99);p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--accum-steps",type=int,default=4)
    p.add_argument("--workers",type=int,default=2);p.add_argument("--lr-merge",type=float,default=1e-4);p.add_argument("--lr-scorer",type=float,default=5e-5)
    p.add_argument("--lr-budget",type=float,default=1e-4);p.add_argument("--lr-backbone",type=float,default=5e-6)
    p.add_argument("--weight-decay",type=float,default=.05);p.add_argument("--val-fraction",type=float,default=.10)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--amp",action="store_true")
    p.add_argument("--output",default="runs/e6_physical_readiness");a=p.parse_args()

    seed_all(a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*88);print("AdaST E5 — FROZEN K(X)+S(X) + CONTENT-ADAPTIVE TOKEN CONSOLIDATION M(X)");print("="*88)
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
    model=AdaSTE6Profile(bb,a.insert_block,a.num_frames,a.temperature)
    load_e5(model,a.e5_checkpoint);model.to(device)

    tp=list(model.temporal_budget.parameters())
    op=[q for n,q in model.named_parameters() if q.requires_grad and not n.startswith("temporal_budget.")]
    opt=torch.optim.AdamW([{"params":tp,"lr":a.lr_budget},{"params":op,"lr":a.lr_backbone}],weight_decay=a.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    scaler=torch.amp.GradScaler("cuda") if a.amp and device.type=="cuda" else None

    # E6 readiness stage: no retraining. Profile the validated E5 policy first.
    bestp=Path(a.e5_checkpoint)
    te_m=final_test(model,tel,device,a.amp,out/"per_video_test.csv")
    dense_patch_tokens=a.num_frames*196
    candidate_tokens=dense_patch_tokens*te_m["combined_retention_mean"]
    candidate_reduction=100.0*(1.0-te_m["combined_retention_mean"])
    summary={"experiment":"AdaST E6 Physical-Execution Readiness Profile",
             "dataset":"UCF101 Split-1","test_top1":te_m["top1"],"test_top5":te_m["top5"],
             "spatial_retention_mean":te_m["r_mean"],"temporal_s_mean":te_m["s_mean"],
             "combined_effective_retention_mean":te_m["combined_retention_mean"],
             "dense_patch_tokens_per_video":dense_patch_tokens,
             "candidate_compact_patch_tokens_mean":candidate_tokens,
             "candidate_token_reduction_pct":candidate_reduction,
             "physical_sequence_reduction":False,
             "note":"Candidate compact-token count only. Vanilla TimeSformer still executes its structured dense sequence."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print("\n"+"="*88)
    print("AdaST E6 — PHYSICAL-EXECUTION READINESS PROFILE")
    print("="*88)
    print(f"Test Top-1                  : {te_m['top1']:.4f}%")
    print(f"Test Top-5                  : {te_m['top5']:.4f}%")
    print(f"Combined effective retention: {te_m['combined_retention_mean']:.4f}")
    print(f"Dense patch tokens/video    : {dense_patch_tokens}")
    print(f"Candidate compact tokens    : {candidate_tokens:.1f}")
    print(f"Candidate token reduction   : {candidate_reduction:.2f}%")
    print("Physical sequence reduction : NO")
    print("Reason: stock TimeSformer downstream blocks require structured space-time token layout.")
    print("Summary:",(out/"summary.json").resolve())
    print("="*88)


if __name__=="__main__":main()
