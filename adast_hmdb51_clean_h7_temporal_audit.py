#!/usr/bin/env python
"""Clean H7: validation-only, no-training audit of H2 and H6 temporal policies."""
import argparse, csv, json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME


def audit(model, loader, device, amp, label):
    rows=[]; captured=[]
    def hook(_module, _inputs, output):
        captured.append(output.detach().float().cpu())
    handle=model.temporal_budget.register_forward_hook(hook)
    model.eval()
    try:
        with torch.inference_mode():
            for i,(x,y,_) in enumerate(loader):
                captured.clear()
                x=x.to(device,non_blocking=True)
                with torch.autocast(device_type='cuda',enabled=amp and device.type=='cuda',dtype=torch.float16):
                    prediction=model(x).logits.float()
                if len(captured)!=1: raise RuntimeError(f'Expected one temporal controller call, got {len(captured)}')
                tlogits=captured[0]
                probs=torch.softmax(tlogits,dim=-1)
                s=model.last_s.detach().float().cpu().reshape(-1)
                expected=model.last_expected_s.detach().float().cpu().reshape(-1)
                pred=prediction.argmax(1).cpu()
                for j in range(len(y)):
                    p=probs[j].tolist(); l=tlogits[j].tolist()
                    rows.append(dict(index=len(rows),label=int(y[j]),prediction=int(pred[j]),correct=int(pred[j]==y[j]),hard_s=int(round(float(s[j]))),expected_s=float(expected[j]),p6=p[0],p7=p[1],p8=p[2],logit6=l[0],logit7=l[1],logit8=l[2],margin_7_vs_8=l[1]-l[2],margin_7_vs_6=l[1]-l[0]))
                if (i+1)%100==0 or i+1==len(loader): print(f'{label}: {len(rows)}/{len(loader.dataset)}',flush=True)
    finally: handle.remove()
    return rows


def summarize(rows):
    return dict(n=len(rows),top1=100*sum(r['correct'] for r in rows)/len(rows),counts={str(k):sum(r['hard_s']==k for r in rows) for k in (6,7,8)},mean_hard_s=float(np.mean([r['hard_s'] for r in rows])),mean_expected_s=float(np.mean([r['expected_s'] for r in rows])),mean_probabilities={f'p{k}':float(np.mean([r[f'p{k}'] for r in rows])) for k in (6,7,8)},mean_logit_margin_7_vs_8=float(np.mean([r['margin_7_vs_8'] for r in rows])),mean_logit_margin_7_vs_6=float(np.mean([r['margin_7_vs_6'] for r in rows])),p7_exceeds_p8=sum(r['p7']>r['p8'] for r in rows),p7_exceeds_p6=sum(r['p7']>r['p6'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('data-root','split-root','dense-checkpoint','h2-checkpoint','h6-epoch1','h6-epoch3','manifest','output'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--split',type=int,default=1);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--amp',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty: '+str(out))
    mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    h1,h1sd=checkpoint(a.dense_checkpoint);h2,h2sd=checkpoint(a.h2_checkpoint)
    for name,obj,protocol in [('H1',h1,'Validation-selected; official test not evaluated'),('H2',h2,'Clean H2 validation-selected; official test not evaluated')]:
        if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:raise RuntimeError(name+' metadata mismatch')
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(h2sd,strict=True)
    for k,v in h1sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('H2 backbone differs from H1: '+k)
    model.to(device).eval()
    loader=DataLoader(HMDB(val,processor,8,False),batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    out.mkdir(parents=True,exist_ok=True)
    checkpoints={'H2':(h2,h2sd),'H6_E1':checkpoint(a.h6_epoch1),'H6_E3':checkpoint(a.h6_epoch3)}
    results={};allrows={}
    for name,(obj,sd) in checkpoints.items():
        if name!='H2':
            if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!='Clean H6 validation-selected; official test not evaluated' or obj.get('epoch')!=int(name[-1]):raise RuntimeError(name+' metadata mismatch')
            for k,v in h2sd.items():
                if not k.startswith('temporal_budget.') and not torch.equal(sd[k].cpu(),v.cpu()):raise RuntimeError(name+' non-temporal parameter changed: '+k)
        model.load_state_dict(sd,strict=True)
        rows=audit(model,loader,device,a.amp,name);allrows[name]=rows;results[name]=summarize(rows)
        with (out/f'{name.lower()}_per_clip.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('RESULT:',json.dumps(dict(checkpoint=name,**results[name])),flush=True)
    comparisons={}
    for name in ('H6_E1','H6_E3'):
        base=allrows['H2'];other=allrows[name]
        comparisons[name]=dict(changed_hard_budget=sum(b['hard_s']!=r['hard_s'] for b,r in zip(base,other)),baseline_correct_to_wrong=sum(b['correct']==1 and r['correct']==0 for b,r in zip(base,other)),baseline_wrong_to_correct=sum(b['correct']==0 and r['correct']==1 for b,r in zip(base,other)),s6_correct=sum(r['correct'] for r in other if r['hard_s']==6),s6_total=sum(r['hard_s']==6 for r in other))
    report=dict(experiment='Clean H7 temporal policy audit',manifest_sha256=digest,validation_n=len(val),official_test_n_not_evaluated=test_n,official_test_evaluated=False,physical_token_deletion=False,results=results,comparisons=comparisons,notes=['No training or checkpoint modifications.','Validation only; policy comparisons are exploratory.','Controller logits captured from temporal_budget forward hook.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPARISONS:',json.dumps(comparisons),flush=True)
    print('CLEAN H7 COMPLETE:',out/'summary.json',flush=True)

if __name__=='__main__':main()
