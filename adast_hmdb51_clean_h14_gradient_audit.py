#!/usr/bin/env python
"""H14: read-only, training-only temporal-controller gradient audit.
No optimizer, no checkpoint writes, no validation/test decoding.
"""
import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def norm_and_flat(grads,params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1).cpu() for g,p in zip(grads,params)])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','split-root','dense-checkpoint','h2-checkpoint','h9b-checkpoint','h13-checkpoint','manifest','profile','output'):
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--split',type=int,default=1)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--per-target',type=int,default=5,help='Distinct training clips per pseudo-target class')
    ap.add_argument('--lambda-target',type=float,default=.2)
    ap.add_argument('--amp',action='store_true')
    a=ap.parse_args()
    if a.per_target<1 or not math.isfinite(a.lambda_target) or a.lambda_target<0:ap.error('Invalid per-target or lambda-target')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):ap.error('Output folder not empty; use a new folder')
    mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    h1,h1sd=checkpoint(a.dense_checkpoint);h2,h2sd=checkpoint(a.h2_checkpoint)
    for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h2,'Clean H2 validation-selected; official test not evaluated')):
        if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:raise RuntimeError('H1/H2 metadata mismatch')
    profile=Path(a.profile);meta=json.loads((profile.parent/'protocol.json').read_text(encoding='utf-8'))
    if meta['manifest_sha256']!=digest or meta['h1_sha256']!=sha(a.dense_checkpoint) or meta['h2_sha256']!=sha(a.h2_checkpoint) or meta['train_n']!=len(train) or meta['seed']!=a.seed:raise RuntimeError('H9-A profile provenance mismatch')
    buckets=defaultdict(list);seen=set()
    with profile.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            i=int(row['index']);s=int(row['target_s'])
            if i in seen or not 0<=i<len(train) or row['path']!=train[i][0] or int(row['label'])!=train[i][1] or s not in (6,7,8):raise RuntimeError('Invalid profile row')
            seen.add(i);buckets[s].append(i)
    if len(seen)!=len(train):raise RuntimeError('Incomplete H9-A profile')
    if any(len(buckets[s])<a.per_target for s in (6,7,8)):raise RuntimeError('Not enough samples for requested per-target')
    selected=[i for s in (6,7,8) for i in buckets[s][:a.per_target]]
    selected_targets={i:s for s in (6,7,8) for i in buckets[s][:a.per_target]}
    states={'H2':h2sd}
    for name,path,protocol in (('H9-B',a.h9b_checkpoint,'Clean H9-B validation-selected; official test not evaluated'),('H13',a.h13_checkpoint,'Clean H13 balanced sampling; official test not evaluated')):
        ck,sd=checkpoint(path)
        if ck.get('manifest_sha256')!=digest or ck.get('profile_sha256')!=sha(profile) or ck.get('protocol')!=protocol or ck.get('split')!=a.split or ck.get('num_frames')!=8 or ck.get('classes')!=classes:raise RuntimeError(name+' metadata mismatch')
        states[name]=sd
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    for p in model.parameters():p.requires_grad_(False)
    for p in model.temporal_budget.parameters():p.requires_grad_(True)
    model.to(device).eval()
    params=list(model.temporal_budget.parameters())
    # HMDB(..., False) reproduces H9-A deterministic input view.
    dataset=HMDB(train,processor,8,False)
    out.mkdir(parents=True,exist_ok=False)
    results={}
    for name,sd in states.items():
        model.load_state_dict(sd,strict=True);model.eval()
        # Ensure all non-controller weights are identical across checkpoints.
        for k,v in model.backbone.state_dict().items():
            if not torch.equal(v.detach().cpu(),h1sd[k].cpu()):raise RuntimeError(name+' backbone differs from H1: '+k)
        records=[]
        for j,i in enumerate(selected,1):
            x,y,path=dataset[i]
            if path!=train[i][0] or int(y)!=train[i][1]:raise RuntimeError('Dataset index mismatch')
            x=x.unsqueeze(0).to(device);y=torch.tensor([int(y)],device=device)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                    logits=model(x).logits
                    probs=model.last_t_probs.float().clamp_min(1e-8)
                    ce=F.cross_entropy(logits.float(),y)
                    target_ce=F.nll_loss(probs.log(),torch.tensor([selected_targets[i]-6],device=device))
                gc=torch.autograd.grad(ce,params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(target_ce,params,allow_unused=True)
            vc=norm_and_flat(gc,params);vt=norm_and_flat(gt,params)
            nc=float(torch.linalg.vector_norm(vc));nt=float(torch.linalg.vector_norm(vt))
            dot=float(torch.dot(vc,vt));cos=dot/(nc*nt) if nc>1e-20 and nt>1e-20 else None
            records.append(dict(index=i,target_s=selected_targets[i],pred_s=int(model.last_s.item()),probs=[float(v) for v in probs[0].detach().cpu()],classification_loss=float(ce.detach()),target_loss=float(target_ce.detach()),ce_grad_norm=nc,target_grad_norm=nt,weighted_target_grad_norm=a.lambda_target*nt,grad_cosine=cos,grad_dot=dot,ce_gradient_nonzero=nc>1e-12,target_gradient_nonzero=nt>1e-12))
            print(f'H14 {name}: {j}/{len(selected)}',flush=True)
            del x,logits,probs,ce,target_ce,gc,gt,vc,vt
        def avg(k,rs):return sum(r[k] for r in rs)/len(rs)
        group={str(s):dict(n=sum(r['target_s']==s for r in records),mean_probs=[sum(r['probs'][q] for r in records if r['target_s']==s)/a.per_target for q in range(3)],mean_ce_grad_norm=avg('ce_grad_norm',[r for r in records if r['target_s']==s]),mean_target_grad_norm=avg('target_grad_norm',[r for r in records if r['target_s']==s])) for s in (6,7,8)}
        valid=[r['grad_cosine'] for r in records if r['grad_cosine'] is not None]
        results[name]=dict(n=len(records),pred_counts={str(s):sum(r['pred_s']==s for r in records) for s in (6,7,8)},mean_ce_grad_norm=avg('ce_grad_norm',records),mean_target_grad_norm=avg('target_grad_norm',records),mean_weighted_target_grad_norm=avg('weighted_target_grad_norm',records),mean_gradient_cosine=(sum(valid)/len(valid) if valid else None),negative_cosine_n=sum(v<0 for v in valid),cosine_defined_n=len(valid),ce_gradient_nonzero_n=sum(r['ce_gradient_nonzero'] for r in records),target_gradient_nonzero_n=sum(r['target_gradient_nonzero'] for r in records),per_target=group,per_clip=records)
        print('H14 MODEL RESULT:',json.dumps({k:v for k,v in results[name].items() if k!='per_clip'}),flush=True)
    report=dict(experiment='Clean H14 read-only temporal controller gradient audit',args=vars(a),manifest_sha256=digest,profile_sha256=sha(profile),train_n=len(train),sampled_train_indices=selected,validation_evaluated=False,official_test_evaluated=False,official_test_n_not_evaluated=test_n,physical_token_deletion=False,important_note='CE gradient uses the scalar expected-S straight-through surrogate, not gradients through hard frame choice/top-k; gradient cosine is a local diagnostic, not causal proof.',models=results)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('H14 COMPLETE:',out/'summary.json',flush=True)

if __name__=='__main__':main()
