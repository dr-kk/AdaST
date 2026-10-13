#!/usr/bin/env python
"""Clean HMDB51 H8: supervised middle-budget feasibility experiment (S=7).
Validation-only selection; official test videos are never decoded. Shape-preserving gating.
Requires local adast_hmdb51_clean_h2_policy.py, adast_e5_adaptive_consolidation.py,
and adast_hmdb51_clean_h1_dense.py.
"""
import argparse, json, csv, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest, evaluate
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--split-root',required=True)
    p.add_argument('--dense-checkpoint',required=True);p.add_argument('--h2-checkpoint',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--split',type=int,default=1)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accum-steps',type=int,default=4);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--lambda-middle',type=float,default=0.2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--amp',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.accum_steps)<1 or a.lambda_middle<0:
        p.error('Invalid epochs/batch/accum or lambda-middle')
    if a.epochs>5: p.error('H8 feasibility experiment limited to 5 epochs')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error(f'Output directory not empty: {out}')
    mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    dense_obj,dense_sd=checkpoint(a.dense_checkpoint)
    if dense_obj.get('manifest_sha256')!=digest or dense_obj.get('classes')!=classes or dense_obj.get('split')!=a.split or dense_obj.get('num_frames')!=8 or dense_obj.get('protocol')!='Validation-selected; official test not evaluated':
        raise RuntimeError('Clean H1 checkpoint metadata/manifest mismatch')
    h2_obj,h2_sd=checkpoint(a.h2_checkpoint)
    if h2_obj.get('manifest_sha256')!=digest or h2_obj.get('classes')!=classes or h2_obj.get('split')!=a.split or h2_obj.get('num_frames')!=8 or h2_obj.get('protocol')!='Clean H2 validation-selected; official test not evaluated':
        raise RuntimeError('Clean H2 checkpoint metadata/manifest mismatch')
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(dense_sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    if model.temporal_levels.tolist()!=[6.0,7.0,8.0]: raise RuntimeError('Unexpected temporal levels ordering')
    model.load_state_dict(h2_sd,strict=True)
    for k,v in dense_sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('H2 backbone differs from clean H1: '+k)
    for q in model.parameters():q.requires_grad_(False)
    for q in model.temporal_budget.parameters():q.requires_grad_(True)
    model.to(device).eval()
    params=[q for q in model.parameters() if q.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=0.01)
    scaler=torch.amp.GradScaler('cuda',enabled=a.amp and device.type=='cuda')
    kw=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    tr_loader=DataLoader(HMDB(train,proc,8,True),shuffle=True,**kw)
    va_loader=DataLoader(HMDB(val,proc,8,False),shuffle=False,**kw)
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(args=vars(a),manifest_sha256=digest,train_n=len(train),val_n=len(val),official_test_n_not_evaluated=test_n,physical_token_deletion=False,trainable='temporal_budget only',objective='video CE plus lambda_middle times negative log probability of S=7; feasibility only'),indent=2),encoding='utf-8')
    print(f'CLEAN H8 | train={len(train)} val={len(val)} official test={test_n} NOT EVALUATED | device={device} | trainable={sum(q.numel() for q in params)}',flush=True)
    baseline=evaluate(model,va_loader,device,a.amp)
    print('BASELINE:',json.dumps(dict(top1=baseline['top1'],s_counts=baseline['temporal_counts'])),flush=True)
    history=[];best_key=None
    for epoch in range(1,a.epochs+1):
        model.eval()  # deterministic hard decisions, no Gumbel noise; ST surrogate remains differentiable
        optimizer.zero_grad(set_to_none=True);n=correct=0;ce_sum=pen_sum=0.;start=time.time()
        for step,(x,y,_) in enumerate(tr_loader,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            group_start=((step-1)//a.accum_steps)*a.accum_steps+1
            group_size=min(a.accum_steps,len(tr_loader)-group_start+1)
            with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                logits=model(x).logits
                ce=F.cross_entropy(logits,y)
                # Exact middle-budget supervision; E5 levels are ordered [6,7,8].
                # H8 is a feasibility test, NOT evidence of per-video adaptivity.
                t_probs=model.last_t_probs.float()
                penalty=-torch.log(t_probs[:,1].clamp_min(1e-8)).mean()
                loss=ce+a.lambda_middle*penalty
            scaler.scale(loss/group_size).backward()
            if step%a.accum_steps==0 or step==len(tr_loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(params,1.0)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            n+=len(y);correct+=int((logits.argmax(1)==y).sum())
            ce_sum+=float(ce.detach())*len(y);pen_sum+=float(penalty.detach())*len(y)
            if step%400==0 or step==len(tr_loader):print(f'E{epoch} step {step}/{len(tr_loader)} train_top1={100*correct/n:.2f}% CE={ce_sum/n:.4f} middle_ce={pen_sum/n:.4f}',flush=True)
        metrics=evaluate(model,va_loader,device,a.amp)
        # Select using validation accuracy first, then prefer fewer actual frames on exact ties.
        # Also save each epoch to allow a transparent validation trade-off comparison.
        hard_s=metrics['temporal_s']['mean'];key=(metrics['top1'],-hard_s)
        row=dict(epoch=epoch,train_top1=100*correct/n,train_ce=ce_sum/n,train_middle_ce=pen_sum/n,val=metrics,seconds=time.time()-start)
        history.append(row)
        ck=dict(model_state_dict=model.state_dict(),epoch=epoch,val=metrics,classes=classes,split=a.split,num_frames=8,manifest_sha256=digest,dense_checkpoint=str(Path(a.dense_checkpoint).resolve()),h2_checkpoint=str(Path(a.h2_checkpoint).resolve()),protocol='Clean H8 validation-selected; official test not evaluated',args=vars(a))
        torch.save(ck,out/f'epoch_{epoch}_val.pt')
        if best_key is None or key>best_key:
            best_key=key;torch.save(ck,out/'best_val.pt')
        print('RESULT:',json.dumps(dict(epoch=epoch,val_top1=metrics['top1'],val_top5=metrics['top5'],hard_s_mean=hard_s,temporal_counts=metrics['temporal_counts'],effective_retention=metrics['combined_effective_retention']['mean'],best_epoch=next(z['epoch'] for z in history if (z['val']['top1'],-z['val']['temporal_s']['mean'])==best_key))),flush=True)
        (out/'summary.json').write_text(json.dumps(dict(experiment='Clean H8 middle-budget feasibility experiment',baseline=baseline,history=history,best_epoch=next(z['epoch'] for z in history if (z['val']['top1'],-z['val']['temporal_s']['mean'])==best_key),official_test_evaluated=False,physical_token_deletion=False),indent=2),encoding='utf-8')
    print('CLEAN H8 COMPLETE:',str(out/'summary.json'),flush=True)

if __name__=='__main__':main()
