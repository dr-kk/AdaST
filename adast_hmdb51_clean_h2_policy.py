#!/usr/bin/env python
"""Clean HMDB51 H2: validation-only adaptation of E5 policy modules, frozen clean H1 backbone.
Requires local adast_e5_adaptive_consolidation.py and adast_hmdb51_clean_h1_dense.py.
No official-test videos are loaded or evaluated. Shape-preserving gating, not physical pruning.
"""
import argparse, csv, hashlib, json, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h1_dense import HMDB, read_hmdb_split, seed_all

POLICY_PREFIXES=('scorer.','budget.','temporal_budget.','consolidator.')
POLICY_BUFFERS=('levels','temporal_levels')

def checkpoint(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(obj,dict): raise RuntimeError('Checkpoint is not a dictionary')
    for key in ('model_state_dict','model','state_dict'):
        if isinstance(obj.get(key),dict): return obj,obj[key]
    raise RuntimeError('Checkpoint has no recognized state dictionary')

def validate_manifest(manifest,manifest_path,split_root,data_root,split):
    raw=manifest_path.read_text(encoding='utf-8').encode('utf-8');digest=hashlib.sha256(raw).hexdigest()
    tr=[(str(p),int(y)) for p,y in manifest['train']]
    va=[(str(p),int(y)) for p,y in manifest['val']]
    official_train,official_test,classes=read_hmdb_split(data_root,split_root,split)
    def key(item):return (str(Path(item[0]).resolve()).casefold(),int(item[1]))
    a,b,c,d=(set(map(key,items)) for items in (tr,va,official_train,official_test))
    if len(a)!=len(tr) or len(b)!=len(va) or a&b or (a|b)!=c or (a|b)&d:
        raise RuntimeError('Manifest does not exactly partition official training videos, or overlaps official test')
    if manifest['classes']!=classes or manifest['split']!=split:raise RuntimeError('Manifest classes/split mismatch')
    if len(tr)!=3213 or len(va)!=357 or len(official_test)!=1530:raise RuntimeError('Unexpected split counts')
    missing=[p for p,_ in tr+va if not Path(p).is_file()]
    if missing:raise FileNotFoundError(f'{len(missing)} manifest video paths unavailable; first: {missing[0]}')
    return tr,va,classes,digest,len(official_test)

def evaluate(model,loader,device,amp):
    model.eval();n=c1=c5=0;loss_sum=0.;r=[];s=[];k=[];joint=[]
    with torch.no_grad():
        for x,y,_ in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast('cuda',enabled=amp and device.type=='cuda',dtype=torch.float16):
                logits=model(x).logits;loss=F.cross_entropy(logits,y)
            n+=len(y);c1+=int((logits.argmax(1)==y).sum());c5+=int((logits.topk(5,1).indices==y[:,None]).any(1).sum())
            loss_sum+=float(loss)*len(y)
            rr=model.last_r.float().cpu().tolist();ss=model.last_s.float().cpu().tolist()
            r.extend(rr);s.extend(ss);k.extend(model.last_k.cpu().tolist());joint.extend([u*v/model.T for u,v in zip(rr,ss)])
    def stats(z):return dict(mean=float(np.mean(z)),std=float(np.std(z)),min=float(np.min(z)),max=float(np.max(z)))
    return dict(n=n,loss=loss_sum/n,top1=100*c1/n,top5=100*c5/n,spatial_retention=stats(r),temporal_s=stats(s),k_per_frame=stats(k),combined_effective_retention=stats(joint),spatial_counts=dict(Counter(map(str,r))),temporal_counts=dict(Counter(map(str,s))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--split-root',required=True)
    p.add_argument('--dense-checkpoint',default=r'runs\hmdb51_clean_h1_dense_s1\best_val.pt')
    p.add_argument('--manifest',default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    p.add_argument('--e5-checkpoint',required=True)
    p.add_argument('--split',type=int,default=1);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accum-steps',type=int,default=4)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--amp',action='store_true')
    p.add_argument('--output',default=r'runs\hmdb51_clean_h2_policy_s1')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.accum_steps)<1: p.error('epochs, batch-size and accum-steps must be positive')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error(f'Output folder not empty: {out}; choose a new output folder')
    manifest_path=Path(a.manifest)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,manifest_path,a.split_root,a.data_root,a.split)
    dense_obj,dense_sd=checkpoint(a.dense_checkpoint)
    if dense_obj.get('manifest_sha256')!=digest:raise RuntimeError('Clean H1 checkpoint manifest hash mismatch; use ORIGINAL local manifest')
    if dense_obj.get('classes')!=classes or dense_obj.get('split')!=a.split or dense_obj.get('num_frames')!=8:raise RuntimeError('Clean H1 class order/split/frames mismatch')
    if dense_obj.get('protocol')!='Validation-selected; official test not evaluated':raise RuntimeError('Dense checkpoint is not marked clean validation-selected')
    e5_obj,e5_sd=checkpoint(a.e5_checkpoint)
    ea=e5_obj.get('args',{})
    if int(ea.get('num_frames',-1))!=8 or int(ea.get('insert_block',-1))!=6:raise RuntimeError('E5 architecture mismatch')
    seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(dense_sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    target=model.state_dict();policy_keys=[k for k in target if k.startswith(POLICY_PREFIXES) or k in POLICY_BUFFERS]
    missing=[k for k in policy_keys if k not in e5_sd or target[k].shape!=e5_sd[k].shape]
    extra=[k for k in e5_sd if k.startswith(POLICY_PREFIXES) and k not in target]
    if missing or extra:raise RuntimeError(f'E5 policy keys mismatch: missing/shape={missing[:12]}, extra={extra[:12]}')
    for key in policy_keys:target[key]=e5_sd[key].clone()
    model.load_state_dict(target,strict=True)
    for key,value in dense_sd.items():
        if not torch.equal(model.backbone.state_dict()[key].cpu(),value.cpu()):raise RuntimeError('Dense backbone overwritten: '+key)
    for param in model.parameters():param.requires_grad=False
    # Hard argmax/topk spatial decisions block classification gradients to scorer/budget.
    # Only temporal surrogate and consolidation are optimized in this CE-only experiment.
    for module in (model.temporal_budget,model.consolidator):
        for param in module.parameters():param.requires_grad=True
    model.to(device).eval()
    trainable=[q for q in model.parameters() if q.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=a.amp and device.type=='cuda')
    kw=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    train_loader=DataLoader(HMDB(train,proc,8,True),shuffle=True,**kw)
    val_loader=DataLoader(HMDB(val,proc,8,False),shuffle=False,**kw)
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(manifest_sha256=digest,train_n=len(train),val_n=len(val),official_test_n_not_evaluated=test_n,dense_checkpoint=str(Path(a.dense_checkpoint).resolve()),e5_checkpoint=str(Path(a.e5_checkpoint).resolve()),physical_token_deletion=False,args=vars(a)),indent=2),encoding='utf-8')
    print(f'CLEAN H2 | train={len(train)} val={len(val)} official test={test_n} NOT EVALUATED | manifest={digest} | device={device}',flush=True)
    print('Frozen backbone and spatial controller; trainable temporal controller + consolidator:',sum(q.numel() for q in trainable),flush=True)
    best=-1.;history=[];best_path=out/'best_val.pt'
    for epoch in range(1,a.epochs+1):
        model.eval();optimizer.zero_grad(set_to_none=True);n=correct=0;loss_sum=0.;start=time.time()
        for step,(x,y,_) in enumerate(train_loader,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            group_start=((step-1)//a.accum_steps)*a.accum_steps+1
            group_size=min(a.accum_steps,len(train_loader)-group_start+1)
            with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                logits=model(x).logits;raw_loss=F.cross_entropy(logits,y)
            scaler.scale(raw_loss/group_size).backward()
            if step%a.accum_steps==0 or step==len(train_loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(trainable,1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            n+=len(y);correct+=int((logits.argmax(1)==y).sum());loss_sum+=float(raw_loss.detach())*len(y)
            if step%200==0 or step==len(train_loader):print(f'E{epoch} step {step}/{len(train_loader)} train_top1={100*correct/n:.2f}% loss={loss_sum/n:.4f}',flush=True)
        metrics=evaluate(model,val_loader,device,a.amp)
        row=dict(epoch=epoch,train_top1=100*correct/n,train_loss=loss_sum/n,val=metrics,seconds=time.time()-start)
        history.append(row)
        print(f'E{epoch}: train={row["train_top1"]:.2f}% val_top1={metrics["top1"]:.2f}% val_top5={metrics["top5"]:.2f}% rho={metrics["spatial_retention"]["mean"]:.4f} S={metrics["temporal_s"]["mean"]:.3f}',flush=True)
        if metrics['top1']>best:
            best=metrics['top1']
            torch.save(dict(model_state_dict=model.state_dict(),epoch=epoch,val=metrics,classes=classes,split=a.split,num_frames=8,manifest_sha256=digest,dense_checkpoint=str(Path(a.dense_checkpoint).resolve()),protocol='Clean H2 validation-selected; official test not evaluated',args=vars(a)),best_path)
            print('Saved validation-selected checkpoint:',best_path,flush=True)
        summary=dict(experiment='Clean HMDB51 H2 frozen-backbone temporal/consolidation adaptation',protocol='Official test NEVER evaluated during training/selection',manifest_sha256=digest,train_n=len(train),val_n=len(val),official_test_n_not_evaluated=test_n,best_val_top1=best,best_epoch=next(z['epoch'] for z in history if z['val']['top1']==best),history=history,physical_token_deletion=False,spatial_controller_trainable=False,args=vars(a))
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['epoch','train_top1','val_top1','val_top5','spatial_retention','temporal_s','combined_retention','seconds'])
            for z in history:w.writerow([z['epoch'],z['train_top1'],z['val']['top1'],z['val']['top5'],z['val']['spatial_retention']['mean'],z['val']['temporal_s']['mean'],z['val']['combined_effective_retention']['mean'],z['seconds']])
    print('CLEAN H2 COMPLETE:',json.dumps(dict(best_val_top1=best,best_checkpoint=str(best_path),test_evaluated=False)),flush=True)

if __name__=='__main__':main()


