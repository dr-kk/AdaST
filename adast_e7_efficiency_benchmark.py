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

    def forward_compact(self, pristine_patch, keep_mask):
        # Physical packed representation. Batch size 1 is required because
        # different clips have different token counts.
        if pristine_patch.size(0) != 1:
            raise RuntimeError("Physical compact mode currently requires --batch-size 1")
        dense = self.forward(pristine_patch, keep_mask)
        idx=torch.nonzero(keep_mask[0] > 0.5,as_tuple=False).squeeze(1)
        return dense[:,idx,:], idx


class CompactTailBlock(nn.Module):
    """Generic self-attention block that accepts arbitrary packed token length."""
    def __init__(self, d=768, heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.n1=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.n2=nn.LayerNorm(d)
        h=int(d*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(h,d),nn.Dropout(dropout))
    def forward(self,x):
        q=self.n1(x)
        a,_=self.attn(q,q,q,need_weights=False)
        x=x+a
        return x+self.mlp(self.n2(x))


class PhysicalCompactTail(nn.Module):
    def __init__(self,d=768,depth=6,heads=12):
        super().__init__()
        self.blocks=nn.ModuleList([CompactTailBlock(d,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(d)
    def forward(self,x):
        for blk in self.blocks:x=blk(x)
        return self.norm(x)

class AdaSTE5(nn.Module):
    def __init__(self,backbone,block=6,T=8,temperature=1.0):
        super().__init__()
        self.backbone=backbone;self.block=block;self.T=T;self.temperature=temperature
        d=backbone.config.hidden_size
        self.scorer=ImportanceScorer(d,256)
        self.budget=DiscreteBudgetNet(d,128,len(LEVELS))
        self.temporal_budget=TemporalBudgetNet(d,128,3)
        self.register_buffer('temporal_levels', torch.tensor([6,7,8], dtype=torch.float32))
        self.consolidator=AdaptiveTokenConsolidator(d=d,proj_dim=96,temperature=0.20)
        self.physical_mode=False
        self.compact_tail=None
        self.last_compact_tokens=None
        self.register_buffer("levels",torch.tensor(LEVELS,dtype=torch.float32))
        self.last_probs=self.last_r=self.last_actual_r=self.last_k=self.last_idx=None
        self.last_t_probs=self.last_s=self.last_t_idx=None
        self.layers()[block].register_forward_pre_hook(self._hook,with_kwargs=True)

    def enable_physical_tail(self):
        # Replace execution from insertion block onward with a generic compact
        # attention tail. Prefix TimeSformer blocks remain pretrained/E5-loaded.
        d=self.backbone.config.hidden_size
        old_layers=self.layers()
        depth=len(old_layers)-self.block
        self.compact_tail=PhysicalCompactTail(d=d,depth=depth,heads=12)
        self.physical_mode=True
        # Hook is used only to capture/compact at the insertion point; the normal
        # full backbone forward is not used in physical mode.
        return depth

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
        if self.physical_mode:
            compact_patch,_compact_idx=self.consolidator.forward_compact(pristine_patch,joint_keep)
            self.last_compact_tokens=int(compact_patch.size(1))
            joint_patch=compact_patch*t_st[:,None,None]
        else:
            joint_patch=self.consolidator(pristine_patch,joint_keep)
            self.last_compact_tokens=int(joint_keep[0].sum().item()) if b==1 else None
            # Straight-through temporal-controller surrogate remains available
            # in shape-preserving E5 mode.
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

    def forward(self,x):
        if not self.physical_mode:
            return self.backbone(pixel_values=x,return_dict=True)

        # We need the pretrained TimeSformer embedding and prefix blocks, then
        # physically packed tokens. Hugging Face versions differ slightly in
        # layer call signatures, so this path uses the model's own embedding
        # module and calls prefix layers with the same common arguments.
        ts=self.backbone.timesformer
        emb_out=ts.embeddings(x)
        h=emb_out
        # Some HF versions return a tuple.
        if isinstance(h,(tuple,list)):h=h[0]

        # Execute only the dense prefix. The registered hook on block `self.block`
        # cannot be used because that block is intentionally skipped.
        for i in range(self.block):
            layer=self.layers()[i]
            try:
                z=layer(h, output_attentions=False)
            except TypeError:
                z=layer(h)
            h=z[0] if isinstance(z,(tuple,list)) else z

        # Apply the exact AdaST K/S/M logic by invoking the hook body directly
        # on the prefix representation. It returns [CLS + physically compact].
        args2,kw2=self._hook(None,(h,),{})
        h=args2[0]
        h=self.compact_tail(h)
        logits=self.backbone.classifier(h[:,0])
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits)



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
    bad=[k for k in z.missing_keys if not k.startswith("compact_tail.")]
    if bad or z.unexpected_keys:
        raise RuntimeError(f"E5 load mismatch: missing={bad[:20]}, unexpected={z.unexpected_keys[:20]}")
    print("Loaded validated E5 K(X)+S(X)+M(X) checkpoint.")


def initialize_compact_tail_from_timesformer(model):
    """Transfer compatible E5/TimeSformer tail parameters into compact tail."""
    copied=[]
    src_layers=model.layers()
    for j,dst in enumerate(model.compact_tail.blocks):
        si=model.block+j
        if si >= len(src_layers): break
        src=src_layers[si]

        for name,dn in [("layernorm_before",dst.n1),("layernorm_after",dst.n2)]:
            sn=getattr(src,name,None)
            if sn is not None and hasattr(sn,"weight") and sn.weight.shape==dn.weight.shape:
                dn.weight.data.copy_(sn.weight.data); dn.bias.data.copy_(sn.bias.data)
                copied.append(f"{si}:{name}")

        inter=getattr(getattr(src,"intermediate",None),"dense",None)
        outdense=getattr(getattr(src,"output",None),"dense",None)
        if inter is not None and outdense is not None:
            if inter.weight.shape==dst.mlp[0].weight.shape:
                dst.mlp[0].weight.data.copy_(inter.weight.data); dst.mlp[0].bias.data.copy_(inter.bias.data)
                copied.append(f"{si}:mlp_fc1")
            if outdense.weight.shape==dst.mlp[3].weight.shape:
                dst.mlp[3].weight.data.copy_(outdense.weight.data); dst.mlp[3].bias.data.copy_(outdense.bias.data)
                copied.append(f"{si}:mlp_fc2")

        att=getattr(src,"attention",None)
        core=getattr(att,"attention",None)
        qkv=getattr(core,"qkv",None)
        proj=getattr(getattr(att,"output",None),"dense",None)
        if qkv is not None and qkv.weight.shape==dst.attn.in_proj_weight.shape:
            dst.attn.in_proj_weight.data.copy_(qkv.weight.data)
            if qkv.bias is not None: dst.attn.in_proj_bias.data.copy_(qkv.bias.data)
            copied.append(f"{si}:attn_qkv")
        if proj is not None and proj.weight.shape==dst.attn.out_proj.weight.shape:
            dst.attn.out_proj.weight.data.copy_(proj.weight.data); dst.attn.out_proj.bias.data.copy_(proj.bias.data)
            copied.append(f"{si}:attn_proj")

    print(f"Transferred {len(copied)} compatible TimeSformer tail groups.")
    if copied: print("Transfer examples:",", ".join(copied[:12]))
    return copied

def configure(model,start=10):
    for p in model.parameters():p.requires_grad=False
    for p in model.compact_tail.parameters():p.requires_grad=True
    for p in model.backbone.classifier.parameters():p.requires_grad=True


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



def make_backbone(num_frames=8,num_labels=101):
    return TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        num_frames=num_frames,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )


def _sync(device):
    if device.type=="cuda": torch.cuda.synchronize()

@torch.no_grad()
def benchmark_model(model,loader,device,amp=True,warmup=20,runs=100):
    import time, numpy as np
    model.eval(); it=iter(loader)
    for _ in range(warmup):
        try:x,_,_=next(it)
        except StopIteration:it=iter(loader);x,_,_=next(it)
        x=x.to(device,non_blocking=True)
        with torch.amp.autocast("cuda",enabled=amp and device.type=="cuda"): _=model(x)
    _sync(device)
    if device.type=="cuda":
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    times=[];compact=[]
    for _ in range(runs):
        try:x,_,_=next(it)
        except StopIteration:it=iter(loader);x,_,_=next(it)
        x=x.to(device,non_blocking=True)
        _sync(device);t0=time.perf_counter()
        with torch.amp.autocast("cuda",enabled=amp and device.type=="cuda"): _=model(x)
        _sync(device);times.append((time.perf_counter()-t0)*1000)
        ct=getattr(model,"last_compact_tokens",None)
        if ct is not None:compact.append(float(ct))
    a=np.asarray(times)
    return {"latency_ms_mean":float(a.mean()),"latency_ms_std":float(a.std()),
            "latency_ms_median":float(np.median(a)),"fps":float(1000/a.mean()),
            "peak_vram_mb":float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=="cuda" else None,
            "compact_tokens_mean":float(np.mean(compact)) if compact else None}

def show(name,r):
    print("\n"+"="*84);print(name);print("="*84)
    print(f"Latency mean   : {r['latency_ms_mean']:.3f} ms/video")
    print(f"Latency median : {r['latency_ms_median']:.3f} ms/video")
    print(f"FPS            : {r['fps']:.3f}")
    print(f"Peak VRAM      : {r['peak_vram_mb']:.2f} MB")
    if r["compact_tokens_mean"] is not None:print(f"Compact tokens : {r['compact_tokens_mean']:.2f}")

def main():
    import argparse,json,os
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True);p.add_argument("--split-root",required=True)
    p.add_argument("--dense-checkpoint",required=True);p.add_argument("--e5-checkpoint",required=True)
    p.add_argument("--e6-checkpoint",required=True)
    p.add_argument("--num-frames",type=int,default=8);p.add_argument("--insert-block",type=int,default=6)
    p.add_argument("--workers",type=int,default=2);p.add_argument("--warmup",type=int,default=20)
    p.add_argument("--runs",type=int,default=100);p.add_argument("--amp",action="store_true")
    p.add_argument("--output",default="runs/e7_efficiency")
    a=p.parse_args();os.makedirs(a.output,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU:",torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")
    _,test_items=read_split(a.data_root,a.split_root,split=1)
    processor=AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    test_ds=UCF(test_items,processor,a.num_frames)
    print(f"Test={len(test_ds)}")
    loader=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=="cuda")
    R={}

    dense=AdaSTE5(make_backbone(a.num_frames,101),block=a.insert_block,T=a.num_frames)
    load_dense(dense.backbone,a.dense_checkpoint);dense.to(device)
    R["dense"]=benchmark_model(dense.backbone,loader,device,a.amp,a.warmup,a.runs);show("DENSE",R["dense"])
    del dense
    if device.type=="cuda":torch.cuda.empty_cache()

    e5=AdaSTE5(make_backbone(a.num_frames,101),block=a.insert_block,T=a.num_frames)
    load_dense(e5.backbone,a.dense_checkpoint);load_e5(e5,a.e5_checkpoint);e5.to(device)
    R["e5"]=benchmark_model(e5,loader,device,a.amp,a.warmup,a.runs);show("E5 SHAPE-PRESERVING",R["e5"])
    del e5
    if device.type=="cuda":torch.cuda.empty_cache()

    e6=AdaSTE5(make_backbone(a.num_frames,101),block=a.insert_block,T=a.num_frames)
    load_dense(e6.backbone,a.dense_checkpoint);e6.enable_physical_tail()
    sd=norm(state(torch.load(a.e6_checkpoint,map_location="cpu",weights_only=False)))
    z=e6.load_state_dict(sd,strict=False)
    if z.missing_keys or z.unexpected_keys:raise RuntimeError(f"E6 load mismatch: {z}")
    e6.to(device)
    R["e6_v2"]=benchmark_model(e6,loader,device,a.amp,a.warmup,a.runs);show("E6-v2 PHYSICAL",R["e6_v2"])
    d,q=R["dense"],R["e6_v2"]
    R["comparison"]={"latency_reduction_pct":100*(d["latency_ms_mean"]-q["latency_ms_mean"])/d["latency_ms_mean"],
                     "speedup_x":d["latency_ms_mean"]/q["latency_ms_mean"],
                     "fps_gain_pct":100*(q["fps"]-d["fps"])/d["fps"],
                     "vram_reduction_pct":100*(d["peak_vram_mb"]-q["peak_vram_mb"])/d["peak_vram_mb"]}
    print("\nE6-v2 vs Dense:",R["comparison"])
    path=os.path.join(a.output,"summary.json")
    with open(path,"w") as f:json.dump(R,f,indent=2)
    print("Summary:",path)

if __name__=="__main__":main()
