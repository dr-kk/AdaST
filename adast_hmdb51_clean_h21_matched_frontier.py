#!/usr/bin/env python
"""Clean H21: matched HMDB51 split-1 validation frontier; no training/test decoding.
Run alongside original adast_hmdb51_clean_h2_policy.py, adast_hmdb51_clean_h1_dense.py,
and adast_e5_adaptive_consolidation.py. Forced budgets override temporal logits only.
"""
import argparse, csv, hashlib, json, time
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def force_logits(module,args,output,idx):
    replacement=torch.full_like(output,-100.0)
    replacement[:,idx]=100.0
    return replacement

def evaluate(model,loader,device,amp,mode):
    hook=None
    if mode!='adaptive':
        s=int(mode[1:]);hook=model.temporal_budget.register_forward_hook(
            lambda module,args,output:force_logits(module,args,output,s-6))
    rows=[];start=time.time()
    try:
        model.eval()
        with torch.inference_mode():
            for i,(x,y,paths) in enumerate(loader):
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                with torch.autocast('cuda',enabled=amp and device.type=='cuda',dtype=torch.float16):
                    logits=model(x).logits
                logp=F.log_softmax(logits.float(),dim=1)
                for j,path in enumerate(paths):
                    yy=int(y[j]);r=float(model.last_r[j]);s=int(model.last_s[j]);k=int(model.last_k[j]);pred=int(logits[j].argmax())
                    top5=bool((logits[j].topk(5).indices==y[j]).any())
                    rows.append(dict(index=i*loader.batch_size+j,path=str(path),label=yy,pred=pred,correct=int(pred==yy),top5=int(top5),nll=float(-logp[j,yy]),spatial_r=r,temporal_s=s,k_per_frame=k,effective_retention=r*s/model.T))
                if (i+1)%100==0 or i+1==len(loader):print(f'H21 {mode}: {i+1}/{len(loader)} batches',flush=True)
    finally:
        if hook is not None:hook.remove()
    n=len(rows)
    return rows,dict(n=n,top1=100*sum(r['correct'] for r in rows)/n,top5=100*sum(r['top5'] for r in rows)/n,
                     mean_nll=sum(r['nll'] for r in rows)/n,mean_spatial_r=sum(r['spatial_r'] for r in rows)/n,
                     mean_temporal_s=sum(r['temporal_s'] for r in rows)/n,
                     mean_effective_retention=sum(r['effective_retention'] for r in rows)/n,
                     temporal_counts=dict(Counter(str(r['temporal_s']) for r in rows)),seconds=time.time()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','split-root','dense-checkpoint','h2-checkpoint','h13-checkpoint','manifest','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--split',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--amp',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output folder not empty; choose a NEW output folder')
    mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    h1,h1sd=checkpoint(a.dense_checkpoint);h2,h2sd=checkpoint(a.h2_checkpoint);h13,h13sd=checkpoint(a.h13_checkpoint)
    expected=[(h1,'Validation-selected; official test not evaluated'),
              (h2,'Clean H2 validation-selected; official test not evaluated'),
              (h13,'Clean H13 balanced sampling; official test not evaluated')]
    for obj,protocol in expected:
        if (obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or
            obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol):
            raise RuntimeError('Checkpoint metadata/protocol mismatch: '+protocol)
    for name,sd in (('H2',h2sd),('H13',h13sd)):
        for k,v in h1sd.items():
            if k not in sd or not torch.equal(sd[k].cpu(),v.cpu()):
                raise RuntimeError(name+' backbone differs from H1: '+k)
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8).to(device).eval()
    kw=dict(batch_size=1,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    loader=DataLoader(HMDB(val,proc,8,False),shuffle=False,**kw)
    out.mkdir(parents=True,exist_ok=False)
    protocol=dict(experiment='Clean H21 matched validation frontier',manifest_sha256=digest,train_n=len(train),val_n=len(val),
                  official_test_n_not_evaluated=test_n,checkpoint_sha256={name:sha(getattr(a,name+'_checkpoint')) for name in ('dense','h2','h13')},
                  args=vars(a),spatial_policy='frozen H2/H13 model-specific spatial policy',
                  forced_temporal='forward hook overrides temporal logits before hard argmax; no parameter modification',
                  validation='same deterministic HMDB(val,proc,8,False), ordered batch=1, each mode',
                  physical_token_deletion=False,measured_speedup=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results={};reference=None
    for model_name,sd in (('h2',h2sd),('h13',h13sd)):
        model.load_state_dict(sd,strict=True)
        for mode in ('S6','S7','S8','adaptive'):
            rows,metrics=evaluate(model,loader,device,a.amp,mode)
            identity=[(r['index'],r['path'],r['label']) for r in rows]
            if reference is None:reference=identity
            elif identity!=reference:raise RuntimeError('Validation video/order mismatch across modes')
            key=model_name+'_'+mode.lower();results[key]=metrics
            with (out/(key+'_per_video.csv')).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (out/'summary.json').write_text(json.dumps(dict(results=results,protocol=protocol,official_test_evaluated=False),indent=2),encoding='utf-8')
            print('H21 MODE:',json.dumps(dict(mode=key,**metrics)),flush=True)
    print('H21 RESULT:',json.dumps(results),flush=True)
    print('CLEAN H21 COMPLETE:',out/'summary.json',flush=True)

if __name__=='__main__':main()
