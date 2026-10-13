#!/usr/bin/env python
"""H2-A exploratory HMDB51 policy-transfer evaluation; no training or checkpoint selection.
Place beside adast_e5_adaptive_consolidation.py and adast_hmdb51_h1_dense_v2.py.
"""
import argparse, csv, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_h1_dense_v2 import HMDB, read_hmdb_split

POLICY_PREFIXES = ('scorer.', 'budget.', 'temporal_budget.', 'consolidator.')
POLICY_BUFFERS = ('levels', 'temporal_levels')

def load_checkpoint(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(obj,dict): raise RuntimeError('Checkpoint must be a dictionary')
    for k in ('model_state_dict','model','state_dict'):
        if k in obj and isinstance(obj[k],dict): return obj,obj[k]
    raise RuntimeError(f'No recognized model state dict in {path}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True);p.add_argument('--split-root',required=True)
    p.add_argument('--dense-checkpoint',required=True);p.add_argument('--e5-checkpoint',required=True)
    p.add_argument('--split',type=int,default=1,choices=(1,2,3));p.add_argument('--num-frames',type=int,default=8)
    p.add_argument('--insert-block',type=int,default=6);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--amp',action='store_true')
    p.add_argument('--output',default='runs/hmdb51_h2a_policy_transfer_s1')
    a=p.parse_args()
    if a.num_frames!=8: raise ValueError('This E5 checkpoint protocol requires 8 frames')
    if a.insert_block!=6: raise ValueError('This E5 checkpoint protocol requires insert block 6')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    _,test,classes=read_hmdb_split(a.data_root,a.split_root,a.split)
    if len(classes)!=51 or len(test)!=1530: raise RuntimeError(f'Unexpected HMDB51 split: classes={len(classes)} test={len(test)}')
    dense_obj,dense_sd=load_checkpoint(a.dense_checkpoint)
    if dense_obj.get('classes')!=classes: raise RuntimeError('Dense checkpoint class ordering differs from official HMDB51 split')
    if dense_obj.get('split')!=a.split or dense_obj.get('num_frames')!=a.num_frames:
        raise RuntimeError('Dense checkpoint split/num_frames mismatch')
    e5_obj,e5_sd=load_checkpoint(a.e5_checkpoint)
    e5_args=e5_obj.get('args',{})
    if int(e5_args.get('num_frames',-1))!=a.num_frames or int(e5_args.get('insert_block',-1))!=a.insert_block:
        raise RuntimeError('E5 checkpoint temporal length / insertion block mismatch')
    if any(k.startswith('module.') for k in dense_sd) or any(k.startswith('module.') for k in e5_sd):
        raise RuntimeError('DataParallel-prefixed checkpoint keys are not supported by this strict loader')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('H2-A | policy-only UCF101 -> HMDB51 transfer |',device,flush=True)
    print('Test videos:',len(test),'| 51 classes | no training',flush=True)
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(dense_sd,strict=True)
    model=AdaSTE5(bb,block=a.insert_block,T=a.num_frames)
    target=model.state_dict()
    keys=[k for k in target if k.startswith(POLICY_PREFIXES) or k in POLICY_BUFFERS]
    if not keys: raise RuntimeError('No adaptive-policy keys found in instantiated model')
    missing=[k for k in keys if k not in e5_sd]
    extra=[k for k in e5_sd if k.startswith(POLICY_PREFIXES) and k not in target]
    mismatched=[(k,tuple(target[k].shape),tuple(e5_sd[k].shape)) for k in keys if k in e5_sd and target[k].shape!=e5_sd[k].shape]
    if missing or extra or mismatched:
        raise RuntimeError(f'Policy mismatch: missing={missing[:12]} extra={extra[:12]} shape={mismatched[:12]}')
    with torch.no_grad():
        for k in keys: target[k].copy_(e5_sd[k])
    model.load_state_dict(target,strict=True)
    # Confirm that the 51-class backbone has not been overwritten by UCF101 weights.
    for k,v in dense_sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):
            raise RuntimeError(f'HMDB51 backbone altered unexpectedly: {k}')
    print(f'Strict loading verified: HMDB51 backbone={len(dense_sd)} tensors; transferred policy={len(keys)} tensors',flush=True)
    model.to(device).eval()
    loader=DataLoader(HMDB(test,proc,a.num_frames,False),batch_size=a.batch_size,shuffle=False,
                      num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    n=c1=c5=0; r=[];ks=[];ss=[];joint=[];gates=[];ent=[];rows=[];t0=time.time()
    with torch.inference_mode():
        for step,(x,y,paths) in enumerate(loader,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                logits=model(x).logits
            if logits.shape[1]!=51: raise RuntimeError('Classifier is not 51-class')
            pred=logits.argmax(1);top5=logits.topk(5,dim=1).indices
            n+=len(y);c1+=int((pred==y).sum());c5+=int((top5==y[:,None]).any(1).sum())
            rv=model.last_r.float().cpu().tolist();kv=model.last_k.cpu().tolist();sv=model.last_s.cpu().tolist()
            gv=model.last_merge_gate.float().cpu().tolist();ev=model.last_merge_entropy.float().cpu().tolist()
            r.extend(rv);ks.extend(kv);ss.extend(sv);gates.extend(gv);ent.extend(ev)
            for i,path in enumerate(paths):
                combined=rv[i]*sv[i]/a.num_frames;joint.append(combined)
                rows.append((path,int(y[i]),int(pred[i]),int(y[i]==pred[i]),rv[i],kv[i],sv[i],combined,gv[i],ev[i]))
            if step%100==0 or step==len(loader):
                print(f'{n}/{len(test)} | Top-1 {100*c1/n:.2f}% | Top-5 {100*c5/n:.2f}% | elapsed {time.time()-t0:.0f}s',flush=True)
    with (out/'per_video_test.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(('video','label','prediction','correct','spatial_r','k_per_frame','temporal_s','combined_effective_retention','merge_gate','merge_entropy'));w.writerows(rows)
    def stats(v):
        z=np.asarray(v,dtype=np.float64)
        return {'mean':float(z.mean()),'std':float(z.std()),'min':float(z.min()),'max':float(z.max())}
    summary={'experiment':'H2-A exploratory UCF101 E5 policy-only transfer onto HMDB51 dense backbone',
             'dataset':'HMDB51','split':a.split,'n_test':n,'num_classes':51,'top1':100*c1/n,'top5':100*c5/n,
             'spatial_retention':stats(r),'k_per_frame':stats(ks),'temporal_s':stats(ss),
             'combined_effective_retention':stats(joint),'merge_gate':stats(gates),'merge_entropy':stats(ent),
             'spatial_counts':dict(Counter(str(v) for v in r)),'temporal_counts':dict(Counter(str(v) for v in ss)),
             'physical_token_deletion':False,'test_selected_dense_checkpoint':True,
             'note':'Policy-only transfer, not zero-shot classification; original dense checkpoint was selected on official test; exploratory results only.',
             'args':vars(a)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('FINAL H2-A:',json.dumps({k:summary[k] for k in ('top1','top5','spatial_retention','temporal_s','combined_effective_retention')},indent=2),flush=True)
    print('Saved:',out/'summary.json',flush=True)

if __name__=='__main__':main()
