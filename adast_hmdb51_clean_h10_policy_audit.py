#!/usr/bin/env python
"""Clean H10: training-only audit of H9-B temporal policy; no optimization/test decoding."""
import argparse, csv, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_hmdb51_clean_h2_policy import checkpoint, validate_manifest
from adast_hmdb51_clean_h1_dense import HMDB, seed_all
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','split-root','dense-checkpoint','h2-checkpoint','h9b-checkpoint','manifest','profile','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--split',type=int,default=1)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--amp',action='store_true')
    a=p.parse_args()
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output folder must be empty; choose a new path')
    mp=Path(a.manifest)
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    h1,h1sd=checkpoint(a.dense_checkpoint)
    h2,h2sd=checkpoint(a.h2_checkpoint)
    h9,h9sd=checkpoint(a.h9b_checkpoint)
    for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h2,'Clean H2 validation-selected; official test not evaluated'),(h9,'Clean H9-B validation-selected; official test not evaluated')):
        if (obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol):
            raise RuntimeError('Checkpoint protocol/manifest mismatch: '+protocol)
    profile=Path(a.profile)
    meta=json.loads((profile.parent/'protocol.json').read_text(encoding='utf-8'))
    if (meta.get('manifest_sha256')!=digest or meta.get('h1_sha256')!=sha(a.dense_checkpoint) or meta.get('h2_sha256')!=sha(a.h2_checkpoint) or meta.get('train_n')!=len(train) or meta.get('seed')!=a.seed):
        raise RuntimeError('H9-A profile provenance mismatch')
    if h9.get('profile_sha256')!=sha(profile):raise RuntimeError('H9-B checkpoint profile hash mismatch')
    targets=[None]*len(train)
    with profile.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            i=int(r['index']);s=int(r['target_s'])
            if not 0<=i<len(train) or targets[i] is not None or r['path']!=train[i][0] or int(r['label'])!=train[i][1] or s not in (6,7,8):
                raise RuntimeError('Invalid training profile row')
            targets[i]=s
    if any(s is None for s in targets):raise RuntimeError('Incomplete profile')
    seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(h9sd,strict=True)
    for k,v in h1sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('H9-B backbone differs from clean H1: '+k)
    model.to(device).eval()
    loader=DataLoader(HMDB(train,proc,8,False),batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    out.mkdir(parents=True,exist_ok=False)
    rows=[];cm={str(t):{str(s):0 for s in (6,7,8)} for t in (6,7,8)}
    captured={}
    def capture(_module,args):
        z=args[0].detach().float()
        appearance=z.mean(dim=(1,2))
        frame=z.mean(dim=2)
        motion=torch.zeros_like(frame)
        motion[:,1:]=frame[:,1:]-frame[:,:-1]
        temporal=motion.abs().mean(dim=1)
        captured['appearance_norm']=float(appearance.norm(dim=1).item())
        captured['motion_norm']=float(temporal.norm(dim=1).item())
        captured['motion_mean']=float(temporal.mean().item())
    hook=model.temporal_budget.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for i,(x,y,paths) in enumerate(loader):
                if paths[0]!=train[i][0] or int(y.item())!=train[i][1]:raise RuntimeError(f'Dataloader alignment mismatch at {i}')
                captured.clear()
                x=x.to(device,non_blocking=True)
                with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                    prediction=model(x).logits
                probs=model.last_t_probs.detach().float().cpu()[0]
                s=int(model.last_s.detach().float().cpu().reshape(-1)[0].item())
                t=targets[i];cm[str(t)][str(s)]+=1
                if 'appearance_norm' not in captured:raise RuntimeError('Temporal feature capture failed')
                p6,p7,p8=(float(z) for z in probs.tolist())
                rows.append(dict(index=i,path=paths[0],label=int(y.item()),target_s=t,pred_s=s,correct=int(prediction.argmax(1).item()==int(y.item())),p6=p6,p7=p7,p8=p8,margin_7_6=math.log(max(p7,1e-12)/max(p6,1e-12)),margin_7_8=math.log(max(p7,1e-12)/max(p8,1e-12)),**captured))
                if (i+1)%200==0 or i+1==len(train):print(f'H10: {i+1}/{len(train)}',flush=True)
    finally:hook.remove()
    fields=list(rows[0])
    with (out/'train_policy_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    group={}
    for t in (6,7,8):
        subset=[r for r in rows if r['target_s']==t]
        group[str(t)]=dict(n=len(subset),pred_counts={str(s):sum(r['pred_s']==s for r in subset) for s in (6,7,8)},mean_probs={f'p{s}':float(np.mean([r[f'p{s}'] for r in subset])) for s in (6,7,8)},mean_appearance_norm=float(np.mean([r['appearance_norm'] for r in subset])),mean_motion_norm=float(np.mean([r['motion_norm'] for r in subset])),mean_motion=float(np.mean([r['motion_mean'] for r in subset])))
    pred_counts=Counter(r['pred_s'] for r in rows)
    result=dict(experiment='Clean H10 training-only H9-B policy audit',train_n=len(rows),val_n_not_evaluated=len(val),official_test_n_not_evaluated=test_n,checkpoint=str(Path(a.h9b_checkpoint).resolve()),checkpoint_epoch=h9.get('epoch'),manifest_sha256=digest,profile_sha256=sha(profile),target_counts={str(s):sum(r['target_s']==s for r in rows) for s in (6,7,8)},pred_counts={str(s):pred_counts[s] for s in (6,7,8)},target_accuracy=100*sum(r['target_s']==r['pred_s'] for r in rows)/len(rows),confusion_target_rows_pred_columns=cm,per_target=group,official_test_evaluated=False,validation_evaluated=False,physical_token_deletion=False,note='Deterministic train-video decoding; H9-A targets were generated with training augmentation, so per-video inputs may differ.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT:',json.dumps({k:result[k] for k in ('train_n','target_counts','pred_counts','target_accuracy','confusion_target_rows_pred_columns','per_target')}),flush=True)
    print('CLEAN H10 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
