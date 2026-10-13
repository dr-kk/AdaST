#!/usr/bin/env python
"""Clean H16: training-only H9-A pseudo-target reliability under repeated temporal sampling.
Uses H2 frozen weights; no training, validation or official-test decoding. Resumable CSV.
Augmented view means randomized temporal frame sampling from HMDB(train=True), NOT spatial augmentation.
"""
import argparse,csv,hashlib,json,random,math
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor,TimesformerForVideoClassification
from adast_hmdb51_clean_h1_dense import HMDB,seed_all
from adast_hmdb51_clean_h2_policy import checkpoint,validate_manifest
from adast_e5_adaptive_consolidation import AdaSTE5,MODEL_NAME
from adast_hmdb51_clean_h9a_train_budget_profile import target

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data-root','split-root','dense-checkpoint','h2-checkpoint','manifest','profile','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--split',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--per-target',type=int,default=100);p.add_argument('--views',type=int,default=3)
    p.add_argument('--amp',action='store_true');a=p.parse_args()
    if a.per_target<1 or a.views<2:p.error('per-target>=1 and views>=2 required')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest_path=Path(a.manifest);manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    train,val,classes,digest,test_n=validate_manifest(manifest,manifest_path,a.split_root,a.data_root,a.split)
    h1,h1sd=checkpoint(a.dense_checkpoint);h2,h2sd=checkpoint(a.h2_checkpoint)
    for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h2,'Clean H2 validation-selected; official test not evaluated')):
        if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:raise RuntimeError('Checkpoint metadata mismatch')
    profile=Path(a.profile);meta=json.loads((profile.parent/'protocol.json').read_text(encoding='utf-8'))
    if meta.get('manifest_sha256')!=digest or meta.get('h1_sha256')!=sha(a.dense_checkpoint) or meta.get('h2_sha256')!=sha(a.h2_checkpoint) or meta.get('train_n')!=len(train) or meta.get('seed')!=a.seed:raise RuntimeError('H9-A provenance mismatch')
    original={}
    with profile.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            i=int(row['index']);s=int(row['target_s'])
            if i in original or not 0<=i<len(train) or row['path']!=train[i][0] or int(row['label'])!=train[i][1] or s not in (6,7,8):raise RuntimeError('Invalid H9-A row')
            original[i]=row
    if len(original)!=len(train):raise RuntimeError('Incomplete H9-A profile')
    tol=float(meta['loss_tolerance']);rng=random.Random(a.seed);groups={s:[] for s in (6,7,8)}
    for i,r in original.items():groups[int(r['target_s'])].append(i)
    chosen=[]
    for s in (6,7,8):
        pool=sorted(groups[s]);chosen+=rng.sample(pool,min(a.per_target,len(pool)))
    chosen=sorted(chosen)
    identity=dict(args=vars(a),manifest_sha256=digest,profile_sha256=sha(profile),h1_sha256=sha(a.dense_checkpoint),h2_sha256=sha(a.h2_checkpoint),selected_indices=chosen,loss_tolerance=tol,train_n=len(train),val_n_not_evaluated=len(val),official_test_n_not_evaluated=test_n,augmentation='random temporal frame sampling only; deterministic per video/view seed',physical_token_deletion=False)
    protocol=out/'protocol.json';csvpath=out/'view_targets.csv'
    if protocol.exists():
        if json.loads(protocol.read_text(encoding='utf-8'))!=identity:raise RuntimeError('Output protocol mismatch; use new folder')
    elif any(out.iterdir()):raise RuntimeError('Output folder not empty; use new folder')
    else:protocol.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    fields=['index','path','label','original_target','view','seed','target_s','target_reason','loss_s6','loss_s7','loss_s8','correct_s6','correct_s7','correct_s8','pred_s6','pred_s7','pred_s8']
    completed={}
    if csvpath.exists():
        with csvpath.open(newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                key=(int(r['index']),int(r['view']))
                if key in completed or key[0] not in chosen or not 1<=key[1]<=a.views or r['path']!=train[key[0]][0] or int(r['label'])!=train[key[0]][1]:raise RuntimeError('Invalid resume CSV')
                completed[key]=r
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    bb.load_state_dict(h1sd,strict=True);model=AdaSTE5(bb,block=6,T=8);model.load_state_dict(h2sd,strict=True)
    for k,v in h1sd.items():
        if not torch.equal(model.backbone.state_dict()[k].cpu(),v.cpu()):raise RuntimeError('H2 backbone differs from H1: '+k)
    model.to(device).eval();chosen_s=[8]
    def forced(z):
        logits=torch.full((z.shape[0],3),-100.,device=z.device,dtype=z.dtype)
        logits[:,chosen_s[0]-6]=100.
        return logits
    original_forward=model.temporal_budget.forward;model.temporal_budget.forward=forced
    ds=HMDB(train,proc,8,True);total=len(chosen)*a.views
    print(f'H16 START | selected={len(chosen)} views={a.views} remaining={total-len(completed)} | val/test NOT EVALUATED',flush=True)
    try:
        with csvpath.open('a',newline='',encoding='utf-8') as f,torch.inference_mode():
            w=csv.DictWriter(f,fieldnames=fields)
            if not completed:w.writeheader();f.flush()
            done=len(completed)
            for i in chosen:
                for view in range(1,a.views+1):
                    if (i,view) in completed:continue
                    view_seed=a.seed*100000000+i*1000+view
                    random.seed(view_seed);torch.manual_seed(view_seed)
                    x,y,path=ds[i]
                    if path!=train[i][0] or int(y)!=train[i][1]:raise RuntimeError('Dataset index mismatch')
                    x=x.unsqueeze(0).to(device);y=y.view(1).to(device)
                    row=dict(index=i,path=path,label=int(y.item()),original_target=int(original[i]['target_s']),view=view,seed=view_seed)
                    for s in (6,7,8):
                        chosen_s[0]=s
                        with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):logits=model(x).logits
                        loss=float(F.cross_entropy(logits.float(),y).item());pred=int(logits.argmax(1).item())
                        if not math.isfinite(loss) or float(model.last_s[0])!=s:raise RuntimeError('Nonfinite loss or forced budget failed')
                        row.update({f'loss_s{s}':loss,f'correct_s{s}':int(pred==int(y.item())),f'pred_s{s}':pred})
                    row['target_s'],row['target_reason']=target(row,tol)
                    w.writerow(row);f.flush();completed[(i,view)]={k:str(v) for k,v in row.items()};done+=1
                    if done%50==0 or done==total:print(f'H16 VIEWS: {done}/{total}',flush=True)
    finally:model.temporal_budget.forward=original_forward
    if len(completed)!=total:raise RuntimeError('Incomplete views')
    by_original={s:dict(n=0,all_views_agree=0,all_views_match_original=0,majority_matches_original=0,view_matches=0,view_total=0,view_target_counts=Counter()) for s in (6,7,8)}
    all_agree=all_match=majority_match=matches=0;transitions=Counter();safe_all=0
    for i in chosen:
        orig=int(original[i]['target_s']);rows=[completed[(i,v)] for v in range(1,a.views+1)]
        ts=[int(r['target_s']) for r in rows];cnt=Counter(ts);ag=len(cnt)==1;match=all(t==orig for t in ts);majority=cnt[orig]>a.views/2
        group=by_original[orig];group['n']+=1;group['all_views_agree']+=ag;group['all_views_match_original']+=match;group['majority_matches_original']+=majority;group['view_matches']+=cnt[orig];group['view_total']+=a.views;group['view_target_counts'].update(ts)
        all_agree+=ag;all_match+=match;majority_match+=majority;matches+=cnt[orig]
        safe_all+=all(int(r['correct_s8'])==1 for r in rows)
        for t in ts:transitions[f'{orig}->{t}']+=1
    summary=dict(experiment='Clean H16 training-only multi-view target reliability',selected_n=len(chosen),views_per_video=a.views,view_total=total,all_views_agree_n=all_agree,all_views_agree_percent=100*all_agree/len(chosen),all_views_match_original_n=all_match,all_views_match_original_percent=100*all_match/len(chosen),majority_matches_original_n=majority_match,view_matches_original_n=matches,view_matches_original_percent=100*matches/total,s8_correct_all_views_n=safe_all,by_original_target={str(k):{**v,'view_target_counts':dict(v['view_target_counts'])} for k,v in by_original.items()},transitions=dict(transitions),official_test_evaluated=False,validation_evaluated=False,notes='Sample stratified by original H9-A pseudo-target, not population-representative without reweighting. Multiple random temporal frame samples only, no spatial augmentation. S8-wrong fallback is not a safe target. Shape-preserving gating.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('H16 RESULT:',json.dumps(summary),flush=True)
    print('CLEAN H16 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
