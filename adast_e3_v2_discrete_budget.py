#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaST E3-v2 — Discrete Content-Adaptive Token Budget
TimeSformer / UCF101 Split-1

Purpose
-------
E2-v2 uses learned per-frame importance with a fixed 50% retention.
E3-v1 collapsed to K=98 for every clip because a batch-size-1 budget penalty
effectively forced each sample toward rho=0.50.

E3-v2 corrects that failure:
  1. Learned controller chooses among discrete retention levels:
       {0.25, 0.375, 0.50, 0.625, 0.75}
  2. Hard forward allocation uses one level per video.
  3. Straight-through Gumbel-Softmax trains the controller.
  4. Mean-compute control uses a running EMA, not a per-video rho=0.50 penalty.
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
LEVELS = (0.25, 0.375, 0.50, 0.625, 0.75)


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


class AdaSTE3V2(nn.Module):
    def __init__(self,backbone,block=6,T=8,temperature=1.0):
        super().__init__()
        self.backbone=backbone;self.block=block;self.T=T;self.temperature=temperature
        d=backbone.config.hidden_size
        self.scorer=ImportanceScorer(d,256)
        self.budget=DiscreteBudgetNet(d,128,len(LEVELS))
        self.register_buffer("levels",torch.tensor(LEVELS,dtype=torch.float32))
        self.last_probs=self.last_r=self.last_actual_r=self.last_k=self.last_idx=None
        self.layers()[block].register_forward_pre_hook(self._hook,with_kwargs=True)

    def layers(self):return self.backbone.timesformer.encoder.layer

    def _hook(self,module,args,kwargs):
        if args:h,rest,src=args[0],args[1:],"args"
        else:h,rest,src=kwargs["hidden_states"],(),"kwargs"
        cls,patch=h[:,:1],h[:,1:];b,m,d=patch.shape
        if m%self.T:raise RuntimeError(f"{m} patch tokens not divisible by T={self.T}")
        n=m//self.T
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


def load_e2(model,path):
    sd=norm(state(torch.load(path,map_location="cpu",weights_only=False)))
    # E2-v2 has backbone.* and scorer.*; budget.* is new.
    filtered={k:v for k,v in sd.items() if not k.startswith("budget.")}
    z=model.load_state_dict(filtered,strict=False)
    bad=[k for k in z.missing_keys if not k.startswith("budget.") and k!="levels"]
    print(f"E2-v2 -> E3-v2: missing={len(z.missing_keys)}, unexpected={len(z.unexpected_keys)}")
    if bad:raise RuntimeError("Non-budget missing keys: "+str(bad[:10]))


def configure(model,start=6):
    for p in model.backbone.parameters():p.requires_grad=False
    for p in model.scorer.parameters():p.requires_grad=True
    for p in model.budget.parameters():p.requires_grad=True
    for i in range(start,len(model.layers())):
        for p in model.layers()[i].parameters():p.requires_grad=True
    for p in model.backbone.classifier.parameters():p.requires_grad=True
    for p in model.backbone.timesformer.layernorm.parameters():p.requires_grad=True


def epoch(model,loader,device,amp,opt=None,scaler=None,accum=4,target=.5,
          lam_budget=2.0,lam_div=.10,ema_decay=.99):
    training=opt is not None;model.train(training);ce=nn.CrossEntropyLoss()
    if training:opt.zero_grad(set_to_none=True)
    total=c1=c5=0;ls=0.;rs=[];rhs=[];ks=[];idxs=[]
    ema=torch.tensor(target,device=device)
    context=torch.enable_grad if training else torch.no_grad
    with context():
        for step,(x,y,_) in enumerate(loader,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
                o=model(x);cl=ce(o.logits,y)
                if training:
                    er=model.last_expected_r.mean()
                    ema=ema_decay*ema+(1-ema_decay)*er
                    budget_loss=(ema-target).pow(2)
                    pbar=model.last_probs.mean(0)
                    # negative entropy: minimizing it favors broad level usage
                    div=(pbar*torch.log(pbar.clamp_min(1e-8))).sum()
                    loss=cl+lam_budget*budget_loss+lam_div*div
                else:loss=cl
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
            rs+=model.last_r.float().cpu().tolist()
            rhs+=model.last_actual_r.float().cpu().tolist()
            ks+=model.last_k.cpu().tolist();idxs+=model.last_idx.cpu().tolist()
            if training and (step%200==0 or step==len(loader)):
                ar=np.asarray(rs)
                print(f"  batch {step}/{len(loader)} | loss {ls/total:.4f} | top1 {100*c1/total:.2f}% "
                      f"| rho {ar.mean():.4f} | rho_std {ar.std():.4f} | levels {dict(Counter(idxs))}")
    r=np.asarray(rs);rh=np.asarray(rhs);k=np.asarray(ks);cnt=Counter(idxs)
    return dict(loss=ls/total,top1=100*c1/total,top5=100*c5/total,
                r_mean=float(r.mean()),r_std=float(r.std()),r_min=float(r.min()),r_max=float(r.max()),
                actual_r_mean=float(rh.mean()),actual_r_std=float(rh.std()),
                k_mean=float(k.mean()),k_std=float(k.std()),k_min=int(k.min()),k_max=int(k.max()),
                level_counts={str(i):int(cnt.get(i,0)) for i in range(len(LEVELS))})


@torch.no_grad()
def final_test(model,loader,device,amp,path):
    model.eval();ce=nn.CrossEntropyLoss();total=c1=c5=0;ls=0
    rs=[];rhs=[];ks=[];idxs=[];rows=[]
    for x,y,paths in loader:
        x=x.to(device);y=y.to(device)
        with torch.autocast("cuda",dtype=torch.float16,enabled=amp and device.type=="cuda"):
            o=model(x);loss=ce(o.logits,y)
        a,b=hits(o.logits,y);bs=y.size(0);total+=bs;c1+=a;c5+=b;ls+=loss.item()*bs
        pred=o.logits.argmax(1).cpu().numpy();yy=y.cpu().numpy()
        rr=model.last_r.cpu().numpy();rh=model.last_actual_r.cpu().numpy()
        kk=model.last_k.cpu().numpy();ii=model.last_idx.cpu().numpy()
        rs+=rr.tolist();rhs+=rh.tolist();ks+=kk.tolist();idxs+=ii.tolist()
        for j in range(bs):
            rows.append([paths[j],int(yy[j]),int(pred[j]),int(pred[j]==yy[j]),
                         int(ii[j]),float(LEVELS[int(ii[j])]),int(kk[j]),float(rh[j])])
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["video","label","prediction","correct","budget_index",
                                    "budget_level","k_per_frame","actual_retention"]);w.writerows(rows)
    r=np.asarray(rs);rh=np.asarray(rhs);k=np.asarray(ks);cnt=Counter(idxs)
    return dict(loss=ls/total,top1=100*c1/total,top5=100*c5/total,r_mean=float(r.mean()),
                r_std=float(r.std()),r_min=float(r.min()),r_max=float(r.max()),
                actual_r_mean=float(rh.mean()),actual_r_std=float(rh.std()),k_mean=float(k.mean()),
                k_std=float(k.std()),k_min=int(k.min()),k_max=int(k.max()),
                level_counts={str(LEVELS[i]):int(cnt.get(i,0)) for i in range(len(LEVELS))})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True);p.add_argument("--split-root",required=True)
    p.add_argument("--dense-checkpoint",required=True);p.add_argument("--e2-checkpoint",required=True)
    p.add_argument("--split",type=int,default=1);p.add_argument("--num-frames",type=int,default=8)
    p.add_argument("--insert-block",type=int,default=6);p.add_argument("--train-from-block",type=int,default=6)
    p.add_argument("--target-rho",type=float,default=.50);p.add_argument("--temperature",type=float,default=1.0)
    p.add_argument("--lambda-budget",type=float,default=2.0);p.add_argument("--lambda-div",type=float,default=.10)
    p.add_argument("--ema-decay",type=float,default=.99);p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=1);p.add_argument("--accum-steps",type=int,default=4)
    p.add_argument("--workers",type=int,default=2);p.add_argument("--lr-scorer",type=float,default=5e-5)
    p.add_argument("--lr-budget",type=float,default=1e-4);p.add_argument("--lr-backbone",type=float,default=5e-6)
    p.add_argument("--weight-decay",type=float,default=.05);p.add_argument("--val-fraction",type=float,default=.10)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--amp",action="store_true")
    p.add_argument("--output",default="runs/e3_v2_discrete_r050");a=p.parse_args()

    seed_all(a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*88);print("AdaST E3-v2 — DISCRETE CONTENT-ADAPTIVE TOKEN BUDGET");print("="*88)
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
    model=AdaSTE3V2(bb,a.insert_block,a.num_frames,a.temperature)
    load_e2(model,a.e2_checkpoint);configure(model,a.train_from_block);model.to(device)

    sp=list(model.scorer.parameters());bp=list(model.budget.parameters())
    op=[q for n,q in model.named_parameters() if q.requires_grad and not n.startswith("scorer.") and not n.startswith("budget.")]
    opt=torch.optim.AdamW([{"params":sp,"lr":a.lr_scorer},{"params":bp,"lr":a.lr_budget},
                           {"params":op,"lr":a.lr_backbone}],weight_decay=a.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    scaler=torch.amp.GradScaler("cuda") if a.amp and device.type=="cuda" else None

    best=-1e9;bestp=out/"best.pt";mp=out/"metrics.csv"
    with mp.open("w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch","train_top1","val_top1","val_top5","val_r_mean","val_r_std",
                                "val_k_mean","val_k_std","level_counts"])
    for ep in range(1,a.epochs+1):
        print(f"\nEPOCH {ep}/{a.epochs}");t=time.time()
        trm=epoch(model,trl,device,a.amp,opt,scaler,a.accum_steps,a.target_rho,
                  a.lambda_budget,a.lambda_div,a.ema_decay)
        vam=epoch(model,val,device,a.amp)
        print(f"Train Top-1={trm['top1']:.2f}%")
        print(f"Val Top-1={vam['top1']:.2f}% | Top-5={vam['top5']:.2f}%")
        print(f"rho={vam['r_mean']:.4f} +/- {vam['r_std']:.4f} | K={vam['k_mean']:.2f} +/- {vam['k_std']:.2f}")
        print("Level counts:",vam["level_counts"]);print(f"Epoch time={time.time()-t:.1f}s")
        with mp.open("a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([ep,trm["top1"],vam["top1"],vam["top5"],vam["r_mean"],
                                    vam["r_std"],vam["k_mean"],vam["k_std"],json.dumps(vam["level_counts"])])
        # Require accuracy and reward genuine adaptation while controlling mean compute.
        score=vam["top1"]-10*abs(vam["actual_r_mean"]-a.target_rho)+min(vam["r_std"],.10)*5
        if score>best:
            best=score;torch.save({"model_state_dict":model.state_dict(),"epoch":ep,"val":vam,"args":vars(a)},bestp)
            print("Saved new best E3-v2 checkpoint.")
        sch.step()

    ck=torch.load(bestp,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model_state_dict"],strict=True);model.to(device)
    te_m=final_test(model,tel,device,a.amp,out/"per_video_test.csv")
    summary={"experiment":"AdaST E3-v2 - Discrete Content-Adaptive Budget","dataset":"UCF101 Split-1",
             "levels":LEVELS,"target_rho":a.target_rho,"dense_reference_top1":92.55,
             "e2_v2_reference_top1":89.92862807295796,"test_top1":te_m["top1"],"test_top5":te_m["top5"],
             "retention_mean":te_m["r_mean"],"retention_std":te_m["r_std"],"retention_min":te_m["r_min"],
             "retention_max":te_m["r_max"],"actual_retention_mean":te_m["actual_r_mean"],
             "k_mean":te_m["k_mean"],"k_std":te_m["k_std"],"k_min":te_m["k_min"],"k_max":te_m["k_max"],
             "level_counts":te_m["level_counts"],"physical_sequence_reduction":False,
             "checkpoint":str(bestp.resolve()),"config":vars(a)}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["top1","top5","rho_mean","rho_std","k_mean","k_std","k_min","k_max","level_counts"])
        w.writerow([te_m["top1"],te_m["top5"],te_m["r_mean"],te_m["r_std"],te_m["k_mean"],te_m["k_std"],
                    te_m["k_min"],te_m["k_max"],json.dumps(te_m["level_counts"])])

    print("\n"+"="*88);print("E3-v2 DISCRETE ADAPTIVE — FINAL TEST RESULT");print("="*88)
    print(f"Test Top-1       : {te_m['top1']:.4f}%");print(f"Test Top-5       : {te_m['top5']:.4f}%")
    print(f"Retention mean   : {te_m['r_mean']:.4f}");print(f"Retention std    : {te_m['r_std']:.4f}")
    print(f"Retention range  : [{te_m['r_min']:.4f}, {te_m['r_max']:.4f}]")
    print(f"K/frame mean/std : {te_m['k_mean']:.2f}/{te_m['k_std']:.2f}")
    print(f"K/frame range    : [{te_m['k_min']},{te_m['k_max']}]")
    print("Level counts     :",te_m["level_counts"])
    print("Physical pruning : NO — shape-preserving E3-v2 ablation")
    print("Summary          :", (out/"summary.json").resolve());print("="*88)


if __name__=="__main__":main()
