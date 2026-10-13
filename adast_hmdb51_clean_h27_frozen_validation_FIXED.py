#!/usr/bin/env python
"""Clean H27: one-time frozen H26 S6->S8 rule validation; never accesses official test."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor,TimesformerForVideoClassification
from adast_hmdb51_clean_h2_policy import checkpoint,validate_manifest
from adast_hmdb51_clean_h1_dense import HMDB,seed_all
from adast_e5_adaptive_consolidation import AdaSTE5,MODEL_NAME

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('data-root','split-root','dense-checkpoint','h13-checkpoint','manifest','frozen-rule','output'):p.add_argument('--'+key,required=True)
 p.add_argument('--split',type=int,default=1);p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,default=2);p.add_argument('--amp',action='store_true')
 a=p.parse_args();out=Path(a.output).resolve()
 if out.exists() and any(out.iterdir()):p.error('Output folder must be empty/new')
 rule=json.loads(Path(a.frozen_rule).read_text(encoding='utf-8'))
 print('H27 RULE KEYS:',sorted(rule),flush=True)
 threshold=float(rule['threshold'])
 if rule.get('signal')!='margin_s6' or rule.get('direction')!='low' or rule.get('comparison')!='<=' or not 0<=threshold<=1:raise RuntimeError('Unexpected frozen rule schema')
 mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
 train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
 class_count = classes if isinstance(classes, int) else len(classes)
 print('H27 SPLIT AUDIT:',json.dumps(dict(train_n=len(train),val_n=len(val),class_count=class_count,classes_type=type(classes).__name__,test_n=test_n)),flush=True)
 if len(val)!=357 or len(train)!=3213 or class_count!=51:raise RuntimeError('Unexpected split counts/classes; inspect H27 SPLIT AUDIT above')
 h1,h1sd=checkpoint(a.dense_checkpoint);h13,h13sd=checkpoint(a.h13_checkpoint)
 for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h13,'Clean H13 balanced sampling; official test not evaluated')):
  if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:raise RuntimeError('Checkpoint protocol mismatch')
 for k,v in h1sd.items():
  key='backbone.'+k
  if key not in h13sd or not torch.equal(h13sd[key].cpu(),v.cpu()):raise RuntimeError('H13 backbone differs from H1: '+key)
 seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
 bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
 bb.load_state_dict(h1sd,strict=True)
 model=AdaSTE5(bb,block=6,T=8).to(device).eval();model.load_state_dict(h13sd,strict=True)
 loader=DataLoader(HMDB(val,proc,8,False),batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
 out.mkdir(parents=True,exist_ok=False)
 def sync():
  if device.type=='cuda':torch.cuda.synchronize()
 def forward(x,mode):
  hook=None
  if mode in (6,8):
   def force(module,args,output):
    z=torch.full_like(output,-100.0);z[:,mode-6]=100.0;return z
   hook=model.temporal_budget.register_forward_hook(force)
  try:
   sync();start=time.perf_counter()
   with torch.inference_mode(),torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):logits=model(x).logits
   sync();elapsed=time.perf_counter()-start
   return logits.float(),elapsed,int(model.last_s[0]),float(model.last_r[0])
  finally:
   if hook is not None:hook.remove()
 rows=[]
 for i,(x,y,paths) in enumerate(loader):
  x=x.to(device,non_blocking=True);label=int(y.item())
  l6,t6,s6,r6=forward(x,6)
  prob=F.softmax(l6,dim=1)[0];top=prob.topk(2).values;margin=float(top[0]-top[1]);escalate=margin<=threshold
  if escalate:l8,t8,s8,r8=forward(x,8);chosen=l8;selected=8
  else:t8=0.0;s8=None;r8=None;chosen=l6;selected=6
  pred6=int(l6.argmax(dim=1).item());pred=int(chosen.argmax(dim=1).item())
  rows.append(dict(index=i,path=str(paths[0]),label=label,s6_pred=pred6,s6_correct=int(pred6==label),margin_s6=margin,escalated=int(escalate),selected_budget=selected,selected_pred=pred,selected_correct=int(pred==label),selected_top5=int(label in chosen[0].topk(5).indices.tolist()),s6_latency_ms=t6*1000,s8_second_pass_latency_ms=t8*1000,total_latency_ms=(t6+t8)*1000,s6_temporal_s=s6,s6_spatial_r=r6,s8_temporal_s=s8,s8_spatial_r=r8))
  if (i+1)%50==0 or i+1==len(loader):print(f'H27 validation: {i+1}/{len(loader)}',flush=True)
 n=len(rows);escalated=sum(r['escalated'] for r in rows)
 summary=dict(experiment='Clean H27 frozen H26 S6->S8 validation',n=n,threshold=threshold,signal='margin_s6',comparison='<=',frozen_rule_sha256=sha(a.frozen_rule),manifest_sha256=digest,checkpoint_sha256=dict(h1=sha(a.dense_checkpoint),h13=sha(a.h13_checkpoint)),s6_top1_pct=100*sum(r['s6_correct'] for r in rows)/n,policy_top1_pct=100*sum(r['selected_correct'] for r in rows)/n,policy_top5_pct=100*sum(r['selected_top5'] for r in rows)/n,escalated_n=escalated,escalated_pct=100*escalated/n,mean_full_model_passes=1+escalated/n,mean_s6_latency_ms=sum(r['s6_latency_ms'] for r in rows)/n,mean_total_policy_latency_ms=sum(r['total_latency_ms'] for r in rows)/n,mean_second_pass_latency_ms_on_escalated=(sum(r['s8_second_pass_latency_ms'] for r in rows)/escalated if escalated else None),official_test_evaluated=False,limitations=['One-time frozen threshold validation; do not retune on validation','S6->S8 escalated videos incur two full forward passes','Latency is sequential model-forward timing with GPU synchronization; excludes data loading/preprocessing','No physical token deletion; effective retention does not imply FLOPs or speedup','Fixed S8/H13 baseline latency not measured in this script; compare with matched protocol separately'])
 with (out/'per_video_validation.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print('H27 RESULT:',json.dumps(summary),flush=True)
 print('CLEAN H27 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
