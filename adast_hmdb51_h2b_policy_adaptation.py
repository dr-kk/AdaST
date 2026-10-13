#!/usr/bin/env python
"""H2-B exploratory HMDB51 policy-only adaptation, train/validation from official training split.
The inherited H1 dense checkpoint was selected using the official test split; this
experiment is exploratory and is NOT a clean held-out test protocol.
Requires adast_e5_adaptive_consolidation.py and adast_hmdb51_h1_dense_v2.py beside this file.
"""
import argparse, csv, json, random, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_h1_dense_v2 import HMDB, read_hmdb_split
from adast_hmdb51_h2a_transfer import load_checkpoint, POLICY_PREFIXES, POLICY_BUFFERS

def stratified(items, fraction, seed):
    groups=defaultdict(list)
    for item in items: groups[item[1]].append(item)
    rng=random.Random(seed); train=[]; val=[]
    for label in sorted(groups):
        group=groups[label][:]; rng.shuffle(group)
        nv=max(1,min(len(group)-1,round(len(group)*fraction)))
        val.extend(group[:nv]); train.extend(group[nv:])
    rng.shuffle(train); rng.shuffle(val)
    if set(x[0] for x in train)&set(x[0] for x in val): raise RuntimeError('Train/val overlap')
    return train,val

def eval_model(model, loader, device, amp, csv_path=None):
    model.eval(); n=c1=c5=0; loss_sum=0.; r=[];s=[];ks=[];combined=[];rows=[]
    with torch.inference_mode():
        for x,y,paths in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast('cuda',enabled=amp and device.type=='cuda',dtype=torch.float16):
                logits=model(x).logits;loss=F.cross_entropy(logits,y)
            pred=logits.argmax(1); top5=logits.topk(5,1).indices
            bs=y.numel();n+=bs;c1+=int((pred==y).sum());c5+=int((top5==y[:,None]).any(1).sum());loss_sum+=float(loss)*bs
            rr=model.last_r.float().cpu().tolist();ss=model.last_s.float().cpu().tolist();kk=model.last_k.cpu().tolist()
            r.extend(rr);s.extend(ss);ks.extend(kk)
            for i,path in enumerate(paths):
                joint=rr[i]*ss[i]/model.T;combined.append(joint)
                if csv_path: rows.append([path,int(y[i]),int(pred[i]),int(pred[i]==y[i]),rr[i],kk[i],ss[i],joint])
    if csv_path:
        with open(csv_path,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['video','label','prediction','correct','spatial_r','k_per_frame','temporal_s','combined_effective_retention']);w.writerows(rows)
    def stats(z):
        z=np.asarray(z,dtype=np.float64)
        return {'mean':float(z.mean()),'std':float(z.std()),'min':float(z.min()),'max':float(z.max())}
    return {'n':n,'loss':loss_sum/n,'top1':100*c1/n,'top5':100*c5/n,
            'spatial_retention':stats(r),'temporal_s':stats(s),'k_per_frame':stats(ks),
            'combined_effective_retention':stats(combined),'spatial_counts':dict(Counter(map(str,r))),
            'temporal_counts':dict(Counter(map(str,s)))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True);p.add_argument('--split-root',required=True)
    p.add_argument('--dense-checkpoint',required=True);p.add_argument('--e5-checkpoint',required=True)
    p.add_argument('--split',type=int,default=1,choices=[1,2,3]);p.add_argument('--num-frames',type=int,default=8)
    p.add_argument('--insert-block',type=int,default=6);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accum-steps',type=int,default=4)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--val-fraction',type=float,default=.10);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--amp',action='store_true');p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--output',default='runs/hmdb51_h2b_policy_only_s1');a=p.parse_args()
    if a.num_frames!=8 or a.insert_block!=6:raise ValueError('E5 requires 8 frames and insert block 6')
    if not 0<a.val_fraction<.5 or a.epochs<1 or a.accum_steps<1:raise ValueError('Invalid validation fraction/epochs/accumulation')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    train_all,test,classes=read_hmdb_split(a.data_root,a.split_root,a.split)
    train,val=stratified(train_all,a.val_fraction,a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'H2-B exploratory policy-only adaptation | train={len(train)} val={len(val)} official_test={len(test)} | device={device}',flush=True)
    dense_obj,dense_sd=load_checkpoint(a.dense_checkpoint)
    e5_obj,e5_sd=load_checkpoint(a.e5_checkpoint)
    if dense_obj.get('classes')!=classes or dense_obj.get('split')!=a.split or dense_obj.get('num_frames')!=a.num_frames:
        raise RuntimeError('Dense checkpoint class order/split/frame mismatch')
    ea=e5_obj.get('args',{})
    if int(ea.get('num_frames',-1))!=8 or int(ea.get('insert_block',-1))!=6:raise RuntimeError('E5 architecture mismatch')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(dense_sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    sd=model.state_dict();policy_keys=[k for k in sd if k.startswith(POLICY_PREFIXES) or k in POLICY_BUFFERS]
    missing=[k for k in policy_keys if k not in e5_sd or sd[k].shape!=e5_sd[k].shape]
    extra=[k for k in e5_sd if k.startswith(POLICY_PREFIXES) and k not in sd]
    if missing or extra:raise RuntimeError(f'Policy mismatch: missing/shape={missing}, extra={extra}')
    for k in policy_keys:sd[k]=e5_sd[k].clone()
    model.load_state_dict(sd,strict=True)
    for k,v in dense_sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('Dense backbone was overwritten: '+k)
    for param in model.parameters():param.requires_grad=False
    for module in (model.scorer,model.budget,model.temporal_budget,model.consolidator):
        for param in module.parameters():param.requires_grad=True
    model.to(device)
    trainable=[param for param in model.parameters() if param.requires_grad]
    print(f'Strict loading verified | policy tensors={len(policy_keys)} | trainable policy params={sum(q.numel() for q in trainable):,} | backbone frozen',flush=True)
    opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=a.amp and device.type=='cuda')
    kw=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    tr=DataLoader(HMDB(train,proc,8,True),shuffle=True,**kw)
    va=DataLoader(HMDB(val,proc,8,False),shuffle=False,**kw)
    te=DataLoader(HMDB(test,proc,8,False),shuffle=False,**kw) if a.evaluate_test else None
    best=-1.;history=[];best_path=out/'best_val.pt'
    for epoch in range(1,a.epochs+1):
        model.eval() # deterministic policies, frozen backbone, stable inference behavior
        opt.zero_grad(set_to_none=True);n=correct=0;loss_total=0.;start=time.time()
        for step,(x,y,_) in enumerate(tr,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                logits=model(x).logits;loss=F.cross_entropy(logits,y)
            scaler.scale(loss/a.accum_steps).backward()
            if step%a.accum_steps==0 or step==len(tr):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(trainable,1.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            n+=y.numel();correct+=int((logits.argmax(1)==y).sum());loss_total+=float(loss.detach())*y.numel()
            if step%200==0 or step==len(tr):
                print(f'E{epoch} step {step}/{len(tr)} train_top1={100*correct/n:.2f}% loss={loss_total/n:.4f}',flush=True)
        metrics=eval_model(model,va,device,a.amp)
        row={'epoch':epoch,'train_top1':100*correct/n,'train_loss':loss_total/n,'val':metrics,'seconds':time.time()-start}
        history.append(row)
        print(f'E{epoch}: train={row["train_top1"]:.2f}% val_top1={metrics["top1"]:.2f}% val_top5={metrics["top5"]:.2f}% r={metrics["spatial_retention"]["mean"]:.4f} S={metrics["temporal_s"]["mean"]:.3f}',flush=True)
        if metrics['top1']>best:
            best=metrics['top1']
            torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'val':metrics,'classes':classes,'split':a.split,'num_frames':8,'args':vars(a),'warning':'Exploratory: inherited dense checkpoint was selected using official test'},best_path)
            print('Saved validation-selected H2-B checkpoint:',best_path,flush=True)
    summary={'experiment':'H2-B HMDB51 policy-only adaptation','train_n':len(train),'val_n':len(val),'test_n':len(test),
             'best_val_top1':best,'history':history,'dense_test_selected':True,
             'physical_token_deletion':False,'note':'Exploratory due to inherited test-selected H1 checkpoint. Test not used for H2-B model selection.', 'args':vars(a)}
    if a.evaluate_test:
        obj,weights=load_checkpoint(best_path);model.load_state_dict(weights,strict=True)
        summary['test']=eval_model(model,te,device,a.amp,out/'per_video_test.csv')
        print('FINAL H2-B:',json.dumps(summary['test'],indent=2),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Saved:',out/'summary.json',flush=True)
if __name__=='__main__':main()
