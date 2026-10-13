#!/usr/bin/env python
"""Clean H9-B: train-only pseudo-target temporal policy, validation-only selection.
Run beside the original clean H1/H2/E5 modules. Official HMDB51 test never decoded.
"""
import argparse, csv, hashlib, json, math, time
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest, evaluate
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME

class TargetHMDB(HMDB):
    def __init__(self, items, processor, targets):
        super().__init__(items, processor, 8, True)
        self.targets=targets
    def __getitem__(self,i):
        x,y,path=super().__getitem__(i)
        return x,y,path,self.targets[i]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','split-root','dense-checkpoint','h2-checkpoint','manifest','profile','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--split',type=int,default=1);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accum-steps',type=int,default=4)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--lambda-target',type=float,default=0.2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--amp',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.accum_steps)<1 or a.lambda_target<0 or not math.isfinite(a.lambda_target):p.error('Invalid training parameters')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output folder not empty; use a new folder')
    mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    h1,h1sd=checkpoint(a.dense_checkpoint);h2,h2sd=checkpoint(a.h2_checkpoint)
    for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h2,'Clean H2 validation-selected; official test not evaluated')):
        if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:raise RuntimeError('Checkpoint metadata mismatch')
    profile=Path(a.profile);profile_meta=json.loads((profile.parent/'protocol.json').read_text(encoding='utf-8'))
    if profile_meta['manifest_sha256']!=digest or profile_meta['h1_sha256']!=sha(a.dense_checkpoint) or profile_meta['h2_sha256']!=sha(a.h2_checkpoint) or profile_meta['train_n']!=len(train) or profile_meta['seed']!=a.seed:raise RuntimeError('H9-A profile/checkpoint/seed mismatch')
    targets=[None]*len(train)
    with profile.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            i=int(row['index']);s=int(row['target_s'])
            if not 0<=i<len(train) or targets[i] is not None or row['path']!=train[i][0] or int(row['label'])!=train[i][1] or s not in (6,7,8):raise RuntimeError('Invalid H9-A CSV row')
            targets[i]=s-6
    if any(x is None for x in targets):raise RuntimeError('Incomplete H9-A profile')
    counts=Counter(targets)
    # Inverse square-root frequency: moderate class balancing; avoids extreme rare-class weights.
    weights=torch.tensor([math.sqrt(len(targets)/(3*counts[i])) for i in range(3)],dtype=torch.float32)
    weights=weights/weights.mean()
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8);model.load_state_dict(h2sd,strict=True)
    for k,v in h1sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('H2 backbone differs from H1: '+k)
    for param in model.parameters():param.requires_grad_(False)
    for param in model.temporal_budget.parameters():param.requires_grad_(True)
    model.to(device).eval();params=[q for q in model.parameters() if q.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=a.amp and device.type=='cuda')
    kw=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    tr_loader=DataLoader(TargetHMDB(train,proc,targets),shuffle=True,**kw)
    va_loader=DataLoader(HMDB(val,proc,8,False),shuffle=False,**kw)
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(args=vars(a),manifest_sha256=digest,profile_sha256=sha(profile),profile_protocol=profile_meta,train_n=len(train),val_n=len(val),official_test_n_not_evaluated=test_n,training_target_counts={str(i+6):counts[i] for i in range(3)},class_weights=weights.tolist(),physical_token_deletion=False,trainable='temporal_budget only'),indent=2),encoding='utf-8')
    print(f'CLEAN H9-B | train={len(train)} val={len(val)} official test={test_n} NOT EVALUATED | device={device} | trainable={sum(q.numel() for q in params)}',flush=True)
    print('TARGETS:',json.dumps({str(i+6):counts[i] for i in range(3)}),'WEIGHTS:',weights.tolist(),flush=True)
    baseline=evaluate(model,va_loader,device,a.amp)
    print('BASELINE:',json.dumps(dict(top1=baseline['top1'],temporal_counts=baseline['temporal_counts'])),flush=True)
    history=[];best_key=None
    for epoch in range(1,a.epochs+1):
        model.eval();optimizer.zero_grad(set_to_none=True)
        n=correct=target_correct=0;ce_sum=target_sum=0.;start=time.time()
        for step,(x,y,paths,t) in enumerate(tr_loader,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);t=t.to(device,dtype=torch.long)
            group_start=((step-1)//a.accum_steps)*a.accum_steps+1
            group_size=min(a.accum_steps,len(tr_loader)-group_start+1)
            with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                logits=model(x).logits
                ce=F.cross_entropy(logits.float(),y)
                # last_t_probs is differentiable and computed by the actual temporal controller.
                probs=model.last_t_probs.float().clamp_min(1e-8)
                target_ce=F.nll_loss(probs.log(),t,weight=weights.to(device))
                loss=ce+a.lambda_target*target_ce
            scaler.scale(loss/group_size).backward()
            if step%a.accum_steps==0 or step==len(tr_loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(params,1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            n+=len(y);correct+=int((logits.argmax(1)==y).sum());target_correct+=int((probs.argmax(1)==t).sum())
            ce_sum+=float(ce.detach())*len(y);target_sum+=float(target_ce.detach())*len(y)
            if step%400==0 or step==len(tr_loader):print(f'E{epoch} step {step}/{len(tr_loader)} train_top1={100*correct/n:.2f}% target_acc={100*target_correct/n:.2f}% CE={ce_sum/n:.4f} target_ce={target_sum/n:.4f}',flush=True)
        metrics=evaluate(model,va_loader,device,a.amp)
        hard_s=metrics['temporal_s']['mean'];key=(metrics['top1'],-hard_s)
        row=dict(epoch=epoch,train_top1=100*correct/n,train_target_accuracy=100*target_correct/n,train_ce=ce_sum/n,train_target_ce=target_sum/n,val=metrics,seconds=time.time()-start)
        history.append(row)
        ck=dict(model_state_dict=model.state_dict(),epoch=epoch,val=metrics,classes=classes,split=a.split,num_frames=8,manifest_sha256=digest,profile_sha256=sha(profile),protocol='Clean H9-B validation-selected; official test not evaluated',args=vars(a))
        torch.save(ck,out/f'epoch_{epoch}_val.pt')
        if best_key is None or key>best_key:best_key=key;torch.save(ck,out/'best_val.pt')
        best_epoch=next(z['epoch'] for z in history if (z['val']['top1'],-z['val']['temporal_s']['mean'])==best_key)
        print('RESULT:',json.dumps(dict(epoch=epoch,val_top1=metrics['top1'],val_top5=metrics['top5'],hard_s_mean=hard_s,temporal_counts=metrics['temporal_counts'],effective_retention=metrics['combined_effective_retention']['mean'],train_target_accuracy=row['train_target_accuracy'],best_epoch=best_epoch)),flush=True)
        (out/'summary.json').write_text(json.dumps(dict(experiment='Clean H9-B train-only pseudo-target policy',baseline=baseline,history=history,best_epoch=best_epoch,official_test_evaluated=False,physical_token_deletion=False),indent=2),encoding='utf-8')
    print('CLEAN H9-B COMPLETE:',out/'summary.json',flush=True)

if __name__=='__main__':main()
