#!/usr/bin/env python
"""Clean H9-A: training-only per-video forced-budget profiling (S=6,7,8).
No training or validation/test decoding. Shape-preserving gating, not physical pruning.
Run alongside the original clean H1/H2/E5 modules. Resume supported by CSV rows.
"""
import argparse, csv, json, hashlib, math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest

FIELDS=['index','path','label','loss_s6','loss_s7','loss_s8','correct_s6','correct_s7','correct_s8','pred_s6','pred_s7','pred_s8','rho_s6','rho_s7','rho_s8','target_s','target_reason']

def target(row,tol):
    # A smaller budget is admissible only if its prediction is correct AND its
    # true-class loss does not exceed the S=8 loss by more than tolerance.
    # If S=8 is wrong, no reliable correctness-preserving pseudo-target exists.
    if not row['correct_s8']:
        return 8,'s8_wrong_no_safe_target'
    for s in (6,7):
        if row[f'correct_s{s}'] and row[f'loss_s{s}'] <= row['loss_s8']+tol:
            return s,'correct_and_within_loss_tolerance'
    return 8,'no_smaller_budget_passed'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--split-root',required=True)
    p.add_argument('--dense-checkpoint',required=True);p.add_argument('--h2-checkpoint',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--split',type=int,default=1)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--loss-tolerance',type=float,default=0.10)
    p.add_argument('--amp',action='store_true');p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.loss_tolerance<0 or not math.isfinite(a.loss_tolerance):p.error('loss-tolerance must be finite and nonnegative')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    h1,h1_sd=checkpoint(a.dense_checkpoint);h2,h2_sd=checkpoint(a.h2_checkpoint)
    for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h2,'Clean H2 validation-selected; official test not evaluated')):
        if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:
            raise RuntimeError('Checkpoint metadata or manifest mismatch')
    identity={'manifest_sha256':digest,'h1_path':str(Path(a.dense_checkpoint).resolve()),'h2_path':str(Path(a.h2_checkpoint).resolve()),'h1_sha256':hashlib.sha256(Path(a.dense_checkpoint).read_bytes()).hexdigest(),'h2_sha256':hashlib.sha256(Path(a.h2_checkpoint).read_bytes()).hexdigest(),'loss_tolerance':a.loss_tolerance,'seed':a.seed,'amp':a.amp,'train_n':len(train),'validation_n_not_evaluated':len(val),'official_test_n_not_evaluated':test_n,'physical_token_deletion':False}
    protocol_path=out/'protocol.json';csv_path=out/'train_budget_profile.csv'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8'))!=identity:raise RuntimeError('Existing output protocol differs; use a NEW output folder')
    elif any(out.iterdir()):raise RuntimeError('Output directory is not empty and has no protocol.json')
    else:protocol_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    completed={}
    if csv_path.exists():
        with csv_path.open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                i=int(row['index'])
                if i in completed or i>=len(train) or i<0 or row['path']!=train[i][0] or int(row['label'])!=train[i][1]:
                    raise RuntimeError('Invalid existing profile CSV; cannot safely resume')
                completed[i]=row
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1_sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8);model.load_state_dict(h2_sd,strict=True)
    for k,v in h1_sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('H2 backbone differs from H1: '+k)
    model.to(device).eval()
    original=model.temporal_budget.forward
    selected=[8]
    def forced_forward(z):
        logits=torch.full((z.shape[0],3),-100.,device=z.device,dtype=z.dtype)
        logits[:,selected[0]-6]=100.
        return logits
    model.temporal_budget.forward=forced_forward
    loader=DataLoader(HMDB(train,proc,8,False),batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    print(f'CLEAN H9-A | train={len(train)} val={len(val)} NOT EVALUATED official test={test_n} NOT EVALUATED | resumed={len(completed)} | device={device}',flush=True)
    try:
        with csv_path.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            if not completed:writer.writeheader();f.flush()
            with torch.inference_mode():
                for i,(x,y,paths) in enumerate(loader):
                    if i in completed:continue
                    if paths[0]!=train[i][0] or int(y[0])!=train[i][1]:raise RuntimeError('Dataset order/path mismatch')
                    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                    row={'index':i,'path':paths[0],'label':int(y[0])}
                    for s in (6,7,8):
                        selected[0]=s
                        with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                            logits=model(x).logits
                        loss=float(F.cross_entropy(logits.float(),y).item())
                        if not math.isfinite(loss):raise RuntimeError(f'Nonfinite loss at index={i}, S={s}')
                        actual=float(model.last_s[0]);pred=int(logits.argmax(1)[0])
                        if actual!=s:raise RuntimeError(f'Forced S={s} failed at index={i}: actual={actual}')
                        row.update({f'loss_s{s}':loss,f'correct_s{s}':int(pred==int(y[0])),f'pred_s{s}':pred,f'rho_s{s}':float(model.last_r[0])*s/8.})
                    row['target_s'],row['target_reason']=target(row,a.loss_tolerance)
                    writer.writerow(row);f.flush()
                    if (i+1)%100==0 or i+1==len(train):print(f'H9-A: {i+1}/{len(train)} target={row["target_s"]}',flush=True)
    finally:model.temporal_budget.forward=original
    rows=[]
    with csv_path.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):rows.append(row)
    if len(rows)!=len(train) or {int(r['index']) for r in rows}!=set(range(len(train))):raise RuntimeError('Incomplete profile; rerun the same command to resume')
    counts={str(s):sum(int(r['target_s'])==s for r in rows) for s in (6,7,8)}
    correctness={str(s):sum(int(r[f'correct_s{s}']) for r in rows) for s in (6,7,8)}
    summary=dict(experiment='Clean H9-A training-only forced temporal budget profiling',protocol=identity,training_target_counts=counts,training_correct_counts=correctness,training_correct_percent={s:100*n/len(train) for s,n in correctness.items()},mean_target_s=sum(int(r['target_s']) for r in rows)/len(rows),no_safe_target_count=sum(r['target_reason']=='s8_wrong_no_safe_target' for r in rows),official_test_evaluated=False,validation_evaluated=False,notes='Pseudo-targets depend on training labels and H2 checkpoint; not proof of generalization. Fixed S=8 fallback for wrong S=8 prediction. Shape-preserving gating only.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('RESULT:',json.dumps({k:summary[k] for k in ('training_target_counts','training_correct_counts','mean_target_s','no_safe_target_count')}),flush=True)
    print('CLEAN H9-A COMPLETE:',out/'summary.json',flush=True)

if __name__=='__main__':main()
