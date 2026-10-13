#!/usr/bin/env python
"""Clean H24: training-only fixed-budget logits/signals; no model training, no val/test decoding.
Run beside original clean H1/H2/E5 modules. All budgets evaluated on identical deterministic train views.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor,TimesformerForVideoClassification
from adast_hmdb51_clean_h2_policy import checkpoint,validate_manifest
from adast_hmdb51_clean_h1_dense import HMDB,seed_all
from adast_e5_adaptive_consolidation import AdaSTE5,MODEL_NAME

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def force(module,args,output,idx):
    result=torch.full_like(output,-100.)
    result[:,idx]=100.
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','split-root','dense-checkpoint','h13-checkpoint','manifest','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--split',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--amp',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output folder nonempty; choose a new output')
    mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    h1,h1sd=checkpoint(a.dense_checkpoint);h13,h13sd=checkpoint(a.h13_checkpoint)
    for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h13,'Clean H13 balanced sampling; official test not evaluated')):
        if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:
            raise RuntimeError('Checkpoint metadata mismatch: '+protocol)
    for key,value in h1sd.items():
        bk='backbone.'+key
        if bk not in h13sd or not torch.equal(value.cpu(),h13sd[bk].cpu()):raise RuntimeError('H13 backbone differs from H1: '+bk)
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8);model.load_state_dict(h13sd,strict=True);model.to(device).eval()
    loader=DataLoader(HMDB(train,processor,8,False),batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    out.mkdir(parents=True,exist_ok=False)
    meta=dict(experiment='Clean H24 train-only fixed-budget inference-time signal extraction',manifest_sha256=digest,train_n=len(train),val_n_not_evaluated=len(val),official_test_n_not_evaluated=test_n,checkpoint_sha256={'h1':sha(a.dense_checkpoint),'h13':sha(a.h13_checkpoint)},args=vars(a),sampling='deterministic HMDB(train, processor, 8, False); identical ordering and views for all budgets',labels_used_for_rule=False,physical_token_deletion=False,measured_speedup=False)
    (out/'protocol.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    identities=None;logits_by_budget={};rows_by_budget={}
    for budget in (6,7,8):
        hook=model.temporal_budget.register_forward_hook(lambda module,args,output,b=budget:force(module,args,output,b-6))
        ids=[];values=[];rows=[];started=time.time()
        try:
            with torch.inference_mode():
                for i,(x,y,paths) in enumerate(loader):
                    x=x.to(device,non_blocking=True)
                    with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                        logits=model(x).logits.float()
                    if logits.shape!=(1,51):raise RuntimeError('Unexpected logits shape')
                    v=logits[0].cpu();probs=F.softmax(v,dim=0);top=probs.topk(2)
                    label=int(y[0]);pred=int(v.argmax());identity=(i,str(paths[0]),label)
                    ids.append(identity);values.append(v.numpy())
                    rows.append(dict(index=i,path=str(paths[0]),label=label,budget=budget,pred=pred,correct=int(pred==label),confidence=float(top.values[0]),margin=float(top.values[0]-top.values[1]),entropy=float(-(probs*probs.clamp_min(1e-12).log()).sum()),spatial_r=float(model.last_r[0]),effective_retention=float(model.last_r[0])*budget/8))
                    if (i+1)%200==0 or i+1==len(loader):print(f'H24 S{budget}: {i+1}/{len(loader)}',flush=True)
        finally:hook.remove()
        if identities is None:identities=ids
        elif ids!=identities:raise RuntimeError('Video identity/order mismatch between budget passes')
        logits_by_budget[budget]=np.stack(values).astype(np.float32);rows_by_budget[budget]=rows
        with (out/f's{budget}_train_signals.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('H24 MODE:',json.dumps(dict(budget=budget,n=len(rows),top1=100*sum(z['correct'] for z in rows)/len(rows),seconds=time.time()-started)),flush=True)
    np.savez_compressed(out/'train_fixed_logits.npz',indices=np.arange(len(train)),labels=np.array([v[2] for v in identities],dtype=np.int64),paths=np.array([v[1] for v in identities]),logits_s6=logits_by_budget[6],logits_s7=logits_by_budget[7],logits_s8=logits_by_budget[8])
    summary=dict(experiment=meta['experiment'],train_n=len(train),validation_evaluated=False,official_test_evaluated=False,top1_by_budget={str(b):100*sum(z['correct'] for z in rows_by_budget[b])/len(train) for b in (6,7,8)},s6_s7_disagreement_n=sum(rows_by_budget[6][i]['pred']!=rows_by_budget[7][i]['pred'] for i in range(len(train))),s6_s8_disagreement_n=sum(rows_by_budget[6][i]['pred']!=rows_by_budget[8][i]['pred'] for i in range(len(train))),outputs=['train_fixed_logits.npz','s6_train_signals.csv','s7_train_signals.csv','s8_train_signals.csv'],limitations=['Training-only deterministic views; no validation/test evaluated','Labels and correctness saved for offline evaluation, never an inference-time input','Full logits stored; no rule trained or selected in H24','Shape-preserving gating; no physical pruning or measured speedup'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('H24 RESULT:',json.dumps(summary),flush=True)
    print('CLEAN H24 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
