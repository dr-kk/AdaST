#!/usr/bin/env python
"""Clean HMDB51 H3: validation-only, no-training diagnostic of H1/H2 checkpoints.
Run from AdaST project directory alongside H1, H2 and E5 Python modules.
All modes are shape-preserving. Official test videos are never decoded/evaluated.
"""
import argparse, csv, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h1_dense import HMDB, read_hmdb_split, seed_all
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest

MODES=('dense','spatial_only','spatial_temporal','full')

def install_diagnostic_hook(model):
    layer=model.layers()[model.block]
    original=model._hook
    found=[]
    for hook_id,fn in list(layer._forward_pre_hooks.items()):
        if getattr(fn,'__self__',None) is model and getattr(fn,'__func__',None) is getattr(original,'__func__',None):
            found.append(hook_id)
    if len(found)!=1:
        raise RuntimeError(f'Expected one AdaSTE5 prehook, found {len(found)}; refusing ambiguous ablation')
    del layer._forward_pre_hooks[found[0]]
    layer._forward_pre_hooks_with_kwargs.pop(found[0], None)
    model.diagnostic_mode='full'

    def hook(module,args,kwargs):
        mode=model.diagnostic_mode
        if mode=='full':
            return original(module,args,kwargs)
        if mode=='dense':
            return args,kwargs
        if args:
            h,rest,src=args[0],args[1:],'args'
        else:
            h,rest,src=kwargs['hidden_states'],(),'kwargs'
        cls,patch=h[:,:1],h[:,1:]
        b,m,d=patch.shape
        if m%model.T:raise RuntimeError('Unexpected token shape')
        n=m//model.T
        scores,delta=model.scorer(patch,model.T)
        logits=model.budget(scores,delta,cls)
        idx=logits.argmax(1)
        r=model.levels[idx]
        k=torch.round(r*n).long().clamp(1,n)
        sf=scores.reshape(b,model.T,n)
        sm=torch.zeros_like(sf)
        for bi in range(b):
            ids=torch.topk(sf[bi],int(k[bi]),dim=1,largest=True,sorted=False).indices
            sm[bi].scatter_(1,ids,1.)
        keep=sm.reshape(b,m)
        model.last_r=r.detach()
        model.last_k=k.detach()
        if mode=='spatial_temporal':
            z=patch.reshape(b,model.T,n,d)
            t_logits=model.temporal_budget(z)
            s=model.temporal_levels[t_logits.argmax(1)]
            frame_feat=z.mean(2)
            novelty=torch.zeros((b,model.T),device=z.device,dtype=z.dtype)
            novelty[:,0]=frame_feat[:,0].pow(2).mean(-1).sqrt()
            novelty[:,1:]=(frame_feat[:,1:]-frame_feat[:,:-1]).pow(2).mean(-1).sqrt()
            tm=torch.zeros_like(novelty)
            for bi in range(b):
                ids=torch.topk(novelty[bi],int(s[bi]),largest=True,sorted=False).indices
                tm[bi].scatter_(0,ids,1.)
            keep=keep*tm[:,:,None].expand(-1,-1,n).reshape(b,m)
        else:
            s=torch.full((b,),model.T,device=patch.device,dtype=r.dtype)
        model.last_s=s.detach()
        gh=torch.cat((cls,patch*keep.unsqueeze(-1)),dim=1)
        if src=='args':return (gh,)+rest,kwargs
        kw=dict(kwargs);kw['hidden_states']=gh
        return args,kw
    layer.register_forward_pre_hook(hook,with_kwargs=True)

@torch.no_grad()
def evaluate(model,loader,mode,device,amp):
    model.eval();model.diagnostic_mode=mode
    n=c1=c5=0;loss_sum=0.;rs=[];ss=[];ks=[]
    for step,(x,y,_) in enumerate(loader,1):
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        with torch.autocast('cuda',enabled=amp and device.type=='cuda',dtype=torch.float16):
            logits=model(x).logits
            loss=F.cross_entropy(logits,y)
        n+=len(y);c1+=int((logits.argmax(1)==y).sum());c5+=int((logits.topk(5,1).indices==y[:,None]).any(1).sum())
        loss_sum+=float(loss)*len(y)
        if mode!='dense':
            rs.extend(model.last_r.float().cpu().tolist())
            ss.extend(model.last_s.float().cpu().tolist())
            ks.extend(model.last_k.float().cpu().tolist())
        if step%100==0 or step==len(loader):print(f'{mode}: {step}/{len(loader)} top1={100*c1/n:.2f}%',flush=True)
    result=dict(mode=mode,n=n,loss=loss_sum/n,top1=100*c1/n,top5=100*c5/n,
                spatial_r_mean=float(np.mean(rs)) if rs else 1.,
                temporal_s_mean=float(np.mean(ss)) if ss else float(model.T),
                k_mean=float(np.mean(ks)) if ks else None,
                temporal_counts={str(int(z)):ss.count(z) for z in sorted(set(ss))} if ss else {},
                physical_token_deletion=False)
    result['effective_retention_mean']=float(np.mean([r*s/model.T for r,s in zip(rs,ss)])) if rs else 1.
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--split-root',required=True)
    p.add_argument('--manifest',default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    p.add_argument('--dense-checkpoint',default=r'runs\hmdb51_clean_h1_dense_s1\best_val.pt')
    p.add_argument('--h2-checkpoint',default=r'runs\hmdb51_clean_h2_policy_s1_v2\best_val.pt')
    p.add_argument('--split',type=int,default=1);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--amp',action='store_true');p.add_argument('--output',default=r'runs\hmdb51_clean_h3_diagnostic_s1')
    a=p.parse_args()
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output folder is not empty; choose a new folder')
    manifest_path=Path(a.manifest)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    _,val,classes,digest,test_n=validate_manifest(manifest,manifest_path,a.split_root,a.data_root,a.split)
    dense_obj,dense_sd=checkpoint(a.dense_checkpoint)
    h2_obj,h2_sd=checkpoint(a.h2_checkpoint)
    if any(obj.get('manifest_sha256')!=digest for obj in (dense_obj,h2_obj)):
        raise RuntimeError('Checkpoint/manifest hash mismatch')
    if dense_obj.get('protocol')!='Validation-selected; official test not evaluated' or h2_obj.get('protocol')!='Clean H2 validation-selected; official test not evaluated':
        raise RuntimeError('Unexpected checkpoint protocol')
    for obj in (dense_obj,h2_obj):
        if obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8:
            raise RuntimeError('Checkpoint class order/split/frame mismatch')
    seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(dense_sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(h2_sd,strict=True)
    for key,value in dense_sd.items():
        if not torch.equal(model.backbone.state_dict()[key].cpu(),value.cpu()):
            raise RuntimeError('H2 backbone differs from clean H1: '+key)
    install_diagnostic_hook(model)
    model.to(device).eval()
    loader=DataLoader(HMDB(val,proc,8,False),batch_size=a.batch_size,shuffle=False,
                      num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    out.mkdir(parents=True,exist_ok=False)
    print(f'CLEAN H3 | val={len(val)} | official test={test_n} NOT EVALUATED | manifest={digest} | device={device}',flush=True)
    results=[]
    for mode in MODES:
        result=evaluate(model,loader,mode,device,a.amp)
        results.append(result)
        print('RESULT:',json.dumps(result),flush=True)
        (out/'summary.json').write_text(json.dumps(dict(experiment='Clean HMDB51 H3 validation-only diagnostic',
            manifest_sha256=digest,validation_n=len(val),official_test_n_not_evaluated=test_n,
            h1_checkpoint=str(Path(a.dense_checkpoint).resolve()),h2_checkpoint=str(Path(a.h2_checkpoint).resolve()),
            modes=results,notes='spatial_only and spatial_temporal zero masked tokens; full uses E5 consolidation; no physical pruning',
            args=vars(a)),indent=2),encoding='utf-8')
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['mode','n','loss','top1','top5','spatial_r_mean','temporal_s_mean','k_mean','effective_retention_mean','physical_token_deletion'],extrasaction='ignore')
        writer.writeheader();writer.writerows(results)
    print('CLEAN H3 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
