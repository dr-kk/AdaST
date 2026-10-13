#!/usr/bin/env python
"""Clean H12: frozen H2 temporal-input feature probe; train-only stratified holdout.
No backbone/policy training; no validation or official-test decoding.
Requires scikit-learn. H9-A labels are deterministic-view pseudo-targets.
"""
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
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
    for name in ('data-root','split-root','dense-checkpoint','h2-checkpoint','manifest','profile','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--split',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--holdout',type=float,default=.25)
    p.add_argument('--amp',action='store_true')
    a=p.parse_args()
    if not 0<a.holdout<1 or a.workers<0:p.error('Invalid holdout/workers')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output folder must be empty; use a new path')
    mp=Path(a.manifest)
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
    h1,h1sd=checkpoint(a.dense_checkpoint)
    h2,h2sd=checkpoint(a.h2_checkpoint)
    for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h2,'Clean H2 validation-selected; official test not evaluated')):
        if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:
            raise RuntimeError('Checkpoint/manifest metadata mismatch')
    profile=Path(a.profile)
    meta=json.loads((profile.parent/'protocol.json').read_text(encoding='utf-8'))
    if (meta.get('manifest_sha256')!=digest or meta.get('h1_sha256')!=sha(a.dense_checkpoint) or meta.get('h2_sha256')!=sha(a.h2_checkpoint) or meta.get('train_n')!=len(train) or meta.get('seed')!=a.seed):
        raise RuntimeError('H9-A profile provenance mismatch')
    targets=[None]*len(train)
    with profile.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            i=int(r['index']);s=int(r['target_s'])
            if not 0<=i<len(train) or targets[i] is not None or r['path']!=train[i][0] or int(r['label'])!=train[i][1] or s not in (6,7,8):
                raise RuntimeError('Invalid H9-A profile row')
            targets[i]=s
    if any(t is None for t in targets):raise RuntimeError('Incomplete H9-A profile')
    seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1sd,strict=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(h2sd,strict=True)
    for k,v in h1sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('H2 backbone differs from H1: '+k)
    model.to(device).eval()
    loader=DataLoader(HMDB(train,proc,8,False),batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(args=vars(a),manifest_sha256=digest,h1_sha256=sha(a.dense_checkpoint),h2_sha256=sha(a.h2_checkpoint),profile_sha256=sha(profile),train_n=len(train),validation_n_not_evaluated=len(val),official_test_n_not_evaluated=test_n,feature_source='frozen H2 temporal_budget input; deterministic train video view',labels='H9-A deterministic training-only pseudo-targets',probe='class-weighted logistic regression on stratified training-only holdout',physical_token_deletion=False),indent=2),encoding='utf-8')
    captured={}
    def capture(_module,args):
        z=args[0].detach().float()  # [B,T,N,D], actual input to temporal controller
        if z.ndim!=4 or z.shape[0]!=1:raise RuntimeError(f'Unexpected temporal input shape: {tuple(z.shape)}')
        frame=z.mean(dim=2)          # [1,T,D]
        appearance=frame.mean(dim=1)
        motion=(frame[:,1:]-frame[:,:-1]).abs().mean(dim=1)
        variability=frame.std(dim=1,unbiased=False)
        feature=torch.cat((appearance,motion,variability),dim=1)
        captured['feature']=feature.squeeze(0).cpu().numpy().astype(np.float32)
    hook=model.temporal_budget.register_forward_pre_hook(capture)
    features=[]
    try:
        with torch.inference_mode():
            for i,(x,y,paths) in enumerate(loader):
                if paths[0]!=train[i][0] or int(y.item())!=train[i][1]:raise RuntimeError(f'Dataset mismatch at {i}')
                captured.clear()
                x=x.to(device,non_blocking=True)
                with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                    model(x)
                if 'feature' not in captured:raise RuntimeError('Feature hook did not fire')
                features.append(captured['feature'])
                if (i+1)%200==0 or i+1==len(train):print(f'H12 FEATURES: {i+1}/{len(train)}',flush=True)
    finally:hook.remove()
    X=np.stack(features);y=np.asarray(targets,dtype=np.int64)
    if not np.isfinite(X).all():raise RuntimeError('Nonfinite features')
    idx=np.arange(len(y))
    fit_idx,hold_idx=train_test_split(idx,test_size=a.holdout,stratify=y,random_state=a.seed)
    scaler=StandardScaler()
    Xfit=scaler.fit_transform(X[fit_idx]);Xhold=scaler.transform(X[hold_idx])
    probe=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=a.seed)
    probe.fit(Xfit,y[fit_idx]);pred=probe.predict(Xhold)
    majority=np.full_like(y[hold_idx],6)
    def metrics(prediction):
        return dict(accuracy_percent=100*accuracy_score(y[hold_idx],prediction),balanced_accuracy_percent=100*balanced_accuracy_score(y[hold_idx],prediction),macro_f1_percent=100*f1_score(y[hold_idx],prediction,labels=[6,7,8],average='macro',zero_division=0),confusion_target_rows_pred_columns=confusion_matrix(y[hold_idx],prediction,labels=[6,7,8]).tolist(),per_class=classification_report(y[hold_idx],prediction,labels=[6,7,8],output_dict=True,zero_division=0))
    result=dict(train_n=len(train),probe_fit_n=len(fit_idx),probe_holdout_n=len(hold_idx),feature_dim=X.shape[1],holdout_target_counts={str(s):int((y[hold_idx]==s).sum()) for s in (6,7,8)},probe=metrics(pred),always_s6_baseline=metrics(majority),validation_evaluated=False,official_test_evaluated=False,notes='Exploratory single-split linear probe; no hyperparameter selection on holdout. Probe performance is not HMDB51 action-classification accuracy or evidence of deployable adaptive policy.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'train_features_and_split.npz',features=X,targets=y,fit_indices=fit_idx,holdout_indices=hold_idx)
    print('H12 RESULT:',json.dumps({k:result[k] for k in ('probe_fit_n','probe_holdout_n','feature_dim','holdout_target_counts','probe','always_s6_baseline')}),flush=True)
    print('CLEAN H12 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
