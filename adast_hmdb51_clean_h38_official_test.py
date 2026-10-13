#!/usr/bin/env python3
"""H38: one-time frozen H13 dense-path evaluation on HMDB51 official split-1 test.
No training, sparse execution, validation-based selection, or checkpoint modification.
"""
import argparse, csv, hashlib, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h1_dense import HMDB, read_hmdb_split

EXPECTED='8b45fc6c63ea8602becd674f59f1f8892174ce352c46fe45a7745423f8405bec'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--split-root',required=True)
    p.add_argument('--h13-checkpoint',default=r'runs\hmdb51_clean_h13_balanced_sampling_s1\best_val.pt')
    p.add_argument('--manifest',default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    p.add_argument('--output',default=r'runs\hmdb51_clean_h38_official_test_s1')
    p.add_argument('--progress-every',type=int,default=50)
    a=p.parse_args()
    if a.progress_every<1:p.error('--progress-every must be positive')
    out=Path(a.output).resolve()
    if out.exists():p.error('Output path already exists; STOP to avoid accidental repeat/overwrite')
    m=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    canonical=hashlib.sha256(json.dumps(m,indent=2,ensure_ascii=False).encode('utf-8')).hexdigest()
    if canonical!=EXPECTED:raise RuntimeError('Canonical manifest hash mismatch; STOP')
    tr=[(str(path),int(y)) for path,y in m['train']]
    val=[(str(path),int(y)) for path,y in m['val']]
    official_train,official_test,classes=read_hmdb_split(a.data_root,a.split_root,1)
    def key(z):return (str(Path(z[0]).resolve()).casefold(),int(z[1]))
    trset,valset,officialset,testset=(set(map(key,z)) for z in (tr,val,official_train,official_test))
    if ((len(tr),len(val),len(official_train),len(official_test))!=(3213,357,3570,1530)
        or any(len(s)!=n for s,n in ((trset,3213),(valset,357),(officialset,3570),(testset,1530)))
        or trset&valset or trset|valset!=officialset or officialset&testset
        or m['classes']!=classes or m['split']!=1):
        raise RuntimeError('Official split/manifest validation failed; STOP')
    if any(not Path(path).is_file() for path,_ in official_test):raise RuntimeError('Missing official test video; STOP')
    ckpath=Path(a.h13_checkpoint)
    ckhash=hashlib.sha256(ckpath.read_bytes()).hexdigest()
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    if (ck.get('manifest_sha256')!=EXPECTED or ck.get('classes')!=classes or ck.get('split')!=1
        or ck.get('num_frames')!=8 or ck.get('protocol')!='Clean H13 balanced sampling; official test not evaluated'):
        raise RuntimeError('H13 checkpoint metadata mismatch; STOP')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.to(device).eval()
    loader=DataLoader(HMDB(official_test,proc,8,False),batch_size=1,shuffle=False,num_workers=0)
    out.mkdir(parents=True,exist_ok=False)
    fields=['index','label','prediction','correct_top1','correct_top5']
    n=top1=top5=0
    start=time.perf_counter()
    with (out/'per_video.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        with torch.inference_mode():
            for ix,(x,y,_) in enumerate(loader):
                logits=model(x.to(device)).logits.detach().float()
                if not torch.isfinite(logits).all():raise RuntimeError(f'Nonfinite logits at test index {ix}; STOP')
                target=int(y.item());pred=int(logits.argmax(1).item())
                hit1=int(pred==target);hit5=int(target in logits.topk(5,dim=1).indices[0].tolist())
                writer.writerow(dict(index=ix,label=target,prediction=pred,correct_top1=hit1,correct_top5=hit5));f.flush()
                n+=1;top1+=hit1;top5+=hit5
                if n%a.progress_every==0 or n==1530:
                    print(f'H38 PROGRESS: {n}/1530 top1={100*top1/n:.2f}% top5={100*top5/n:.2f}%',flush=True)
    if n!=1530:raise RuntimeError(f'Incomplete official test: {n}/1530; STOP')
    result=dict(experiment='H38 frozen H13 dense original official HMDB51 split-1 test',status='PASS',n=n,
                official_test_n=1530,top1_percent=100*top1/n,top5_percent=100*top5/n,
                top1_correct=top1,top5_correct=top5,manifest_canonical_sha256=canonical,
                checkpoint_sha256=ckhash,checkpoint_loaded_strictly=True,device=str(device),amp=False,
                training_performed=False,sparse_path_used=False,validation_evaluated=False,
                official_test_evaluated=True,elapsed_seconds=round(time.perf_counter()-start,2),
                limitations=['Single frozen H13 checkpoint and HMDB51 split-1 official test.',
                             'No test-set tuning or model selection permitted after this evaluation.',
                             'No latency, FLOP reduction or speedup claim established by H38.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('H38 RESULT:',json.dumps(result),flush=True)
    print('H38 COMPLETE:',out,flush=True)
if __name__=='__main__':main()
