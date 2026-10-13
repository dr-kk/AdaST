#!/usr/bin/env python
"""Clean HMDB51 H4: validation-only temporal controller diagnosis; no training or test evaluation.
Run alongside the working H1/H2/E5 modules from the AdaST project directory.
"""
import argparse, csv, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--split-root',required=True)
    p.add_argument('--dense-checkpoint',default=r'runs\hmdb51_clean_h1_dense_s1\best_val.pt')
    p.add_argument('--h2-checkpoint',default=r'runs\hmdb51_clean_h2_policy_s1_v2\best_val.pt')
    p.add_argument('--manifest',default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    p.add_argument('--split',type=int,default=1);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--amp',action='store_true');p.add_argument('--output',default=r'runs\hmdb51_clean_h4_temporal_s1')
    a=p.parse_args()
    if a.batch_size<1 or a.workers<0:p.error('batch-size must be positive; workers must be nonnegative')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error(f'Output folder not empty: {out}; choose a new one')
    mp=Path(a.manifest)
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    dense_obj,dense_sd=checkpoint(a.dense_checkpoint)
    h2_obj,h2_sd=checkpoint(a.h2_checkpoint)
    if dense_obj.get('manifest_sha256')!=digest or h2_obj.get('manifest_sha256')!=digest:
        raise RuntimeError('H1/H2 checkpoint manifest hash mismatch')
    if dense_obj.get('classes')!=classes or h2_obj.get('classes')!=classes:
        raise RuntimeError('Checkpoint class order mismatch')
    if any(obj.get('split')!=a.split or obj.get('num_frames')!=8 for obj in (dense_obj,h2_obj)):
        raise RuntimeError('Checkpoint split or frame count mismatch')
    if dense_obj.get('protocol')!='Validation-selected; official test not evaluated':
        raise RuntimeError('H1 checkpoint does not have expected clean protocol')
    if h2_obj.get('protocol')!='Clean H2 validation-selected; official test not evaluated':
        raise RuntimeError('H2 checkpoint does not have expected clean protocol')
    seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(dense_sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(h2_sd,strict=True)
    if not all(torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()) for k,v in dense_sd.items()):
        raise RuntimeError('H2 backbone differs from clean H1 backbone')
    model.to(device).eval()
    captured=[]
    def capture_temporal_logits(module,inputs,output):
        if not isinstance(output,torch.Tensor) or output.ndim!=2 or output.shape[1]!=3:
            raise RuntimeError('Unexpected temporal controller logits shape')
        captured.append(output.detach().float().cpu())
    handle=model.temporal_budget.register_forward_hook(capture_temporal_logits)
    kw=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    loader=DataLoader(HMDB(val,proc,8,False),shuffle=False,**kw)
    out.mkdir(parents=True,exist_ok=False)
    rows=[];counts=Counter();correct=0;seen=0
    print(f'CLEAN H4 | val={len(val)} official test={test_n} NOT EVALUATED | manifest={digest} | device={device}',flush=True)
    try:
        with torch.inference_mode():
            for x,y,paths in loader:
                captured.clear()
                x=x.to(device,non_blocking=True)
                with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                    pred_logits=model(x).logits
                if len(captured)!=1:raise RuntimeError(f'Expected one temporal-controller call per batch, got {len(captured)}')
                tl=captured[0];probs=torch.softmax(tl,dim=1)
                choice=tl.argmax(1);expected=(probs*torch.tensor([6.,7.,8.])).sum(1)
                actual=model.last_s.detach().cpu().long()
                if not torch.equal(actual,choice+6):raise RuntimeError('Captured temporal logits disagree with model.last_s')
                prediction=pred_logits.argmax(1).detach().cpu();y=y.cpu()
                for i,path in enumerate(paths):
                    q=probs[i].tolist();z=tl[i].tolist();chosen=int(actual[i]);counts[chosen]+=1
                    seen+=1;correct+=int(prediction[i]==y[i])
                    ordered=sorted(q,reverse=True)
                    rows.append(dict(video=str(path),label=int(y[i]),prediction=int(prediction[i]),correct=int(prediction[i]==y[i]),
                        logit_s6=z[0],logit_s7=z[1],logit_s8=z[2],prob_s6=q[0],prob_s7=q[1],prob_s8=q[2],
                        selected_s=chosen,expected_s=float(expected[i]),prob_margin_top1_top2=ordered[0]-ordered[1],
                        entropy_nats=-sum(v*math.log(max(v,1e-12)) for v in q),spatial_r=float(model.last_r[i].detach().cpu())))
                if seen%100<a.batch_size or seen==len(val):
                    print(f'validation {seen}/{len(val)} | top1={100*correct/seen:.2f}% | S-counts={dict(counts)}',flush=True)
    finally:handle.remove()
    if seen!=len(val):raise RuntimeError(f'Incomplete validation: {seen}/{len(val)}')
    with (out/'per_video_temporal.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def stats(key):
        arr=np.array([r[key] for r in rows],dtype=float)
        return dict(mean=float(arr.mean()),std=float(arr.std()),min=float(arr.min()),max=float(arr.max()))
    summary=dict(experiment='Clean H4 temporal-controller diagnosis',protocol='Validation only; official test not evaluated',
        manifest_sha256=digest,validation_n=seen,official_test_n_not_evaluated=test_n,
        validation_top1=100*correct/seen,temporal_counts={str(s):counts[s] for s in (6,7,8)},
        logits={f'S{s}':stats(f'logit_s{s}') for s in (6,7,8)},
        probabilities={f'S{s}':stats(f'prob_s{s}') for s in (6,7,8)},
        expected_s=stats('expected_s'),prob_margin_top1_top2=stats('prob_margin_top1_top2'),
        entropy_nats=stats('entropy_nats'),spatial_r=stats('spatial_r'),
        dense_checkpoint=str(Path(a.dense_checkpoint).resolve()),h2_checkpoint=str(Path(a.h2_checkpoint).resolve()),
        shape_preserving=True,physical_token_deletion=False,args=vars(a))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('CLEAN H4 COMPLETE:',json.dumps(dict(val_top1=summary['validation_top1'],temporal_counts=summary['temporal_counts'],
        mean_probs={k:round(v['mean'],5) for k,v in summary['probabilities'].items()},
        expected_s_mean=summary['expected_s']['mean'],mean_margin=summary['prob_margin_top1_top2']['mean'],
        summary=str(out/'summary.json'),test_evaluated=False)),flush=True)

if __name__=='__main__':main()
