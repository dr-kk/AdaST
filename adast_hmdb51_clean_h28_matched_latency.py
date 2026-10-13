#!/usr/bin/env python
"""H28: matched, validation-only forward latency and paired error audit. No threshold tuning/test."""
import argparse,csv,hashlib,json,time,statistics
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
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('data-root','split-root','dense-checkpoint','h13-checkpoint','manifest','frozen-rule','output'):p.add_argument('--'+k,required=True)
 p.add_argument('--h27-per-video',default=None,help='Optional H27 CSV: assert identical policy decisions and predictions')
 p.add_argument('--split',type=int,default=1);p.add_argument('--seed',type=int,default=42)
 p.add_argument('--workers',type=int,default=2);p.add_argument('--amp',action='store_true')
 a=p.parse_args();out=Path(a.output).resolve()
 if out.exists() and any(out.iterdir()):p.error('Output directory must be new or empty')
 rule=json.loads(Path(a.frozen_rule).read_text(encoding='utf-8'));threshold=float(rule['threshold'])
 if rule.get('signal')!='margin_s6' or rule.get('direction')!='low' or rule.get('comparison')!='<=' or not 0<=threshold<=1:raise RuntimeError('Frozen rule mismatch')
 mp=Path(a.manifest);manifest=json.loads(mp.read_text(encoding='utf-8'))
 train,val,classes,digest,test_n=validate_manifest(manifest,mp,a.split_root,a.data_root,a.split)
 nc=classes if isinstance(classes,int) else len(classes)
 print('H28 SPLIT:',json.dumps(dict(train=len(train),val=len(val),classes=nc,official_test_not_evaluated=test_n)),flush=True)
 if len(train)!=3213 or len(val)!=357 or nc!=51:raise RuntimeError('Unexpected split')
 h1,h1sd=checkpoint(a.dense_checkpoint);h13,h13sd=checkpoint(a.h13_checkpoint)
 for obj,protocol in ((h1,'Validation-selected; official test not evaluated'),(h13,'Clean H13 balanced sampling; official test not evaluated')):
  if obj.get('manifest_sha256')!=digest or obj.get('classes')!=classes or obj.get('split')!=a.split or obj.get('num_frames')!=8 or obj.get('protocol')!=protocol:raise RuntimeError('Checkpoint protocol mismatch')
 for k,v in h1sd.items():
  bk='backbone.'+k
  if bk not in h13sd or not torch.equal(h13sd[bk].cpu(),v.cpu()):raise RuntimeError('Backbone mismatch: '+bk)
 seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
 bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
 bb.load_state_dict(h1sd,strict=True)
 model=AdaSTE5(bb,block=6,T=8).to(device).eval();model.load_state_dict(h13sd,strict=True)
 loader=DataLoader(HMDB(val,proc,8,False),batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
 h27=None
 if a.h27_per_video:
  with open(a.h27_per_video,newline='',encoding='utf-8-sig') as f:h27=list(csv.DictReader(f))
  if len(h27)!=len(val):raise RuntimeError('H27 reference count mismatch')
 out.mkdir(parents=True,exist_ok=True)
 def sync():
  if device.type=='cuda':torch.cuda.synchronize()
 def forward(x,mode):
  hook=None
  if mode!='adaptive':
   budget=int(mode[1:])
   def force(module,args,output):
    z=torch.full_like(output,-100.0);z[:,budget-6]=100.0;return z
   hook=model.temporal_budget.register_forward_hook(force)
  try:
   sync();start=time.perf_counter()
   with torch.inference_mode(),torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):logits=model(x).logits
   sync();ms=(time.perf_counter()-start)*1000
   return logits.float(),ms,int(model.last_s[0]),float(model.last_r[0])
  finally:
   if hook is not None:hook.remove()
 modes=('S6','S7','S8','adaptive');rows=[]
 for i,(x,y,paths) in enumerate(loader):
  x=x.to(device,non_blocking=True);label=int(y.item());path=str(paths[0])
  # Rotate order across videos to reduce systematic warmup/order bias.
  order=modes[i%len(modes):]+modes[:i%len(modes)]
  measured={mode:forward(x,mode) for mode in order}
  l6,t6,s6,r6=measured['S6'];prob=F.softmax(l6[0],dim=0);top=prob.topk(2).values
  margin=float(top[0]-top[1]);escalate=margin<=threshold
  # H27 performs S8 *after* S6 only on escalated videos. Its extra cost must be counted.
  if escalate:second_logits,second_ms,_,_=forward(x,'S8')
  else:second_logits=None;second_ms=0.0
  selected=second_logits if escalate else l6
  preds={mode:int(measured[mode][0].argmax(dim=1).item()) for mode in modes}
  selected_pred=int(selected.argmax(dim=1).item())
  row=dict(index=i,path=path,label=label,margin_s6=margin,escalated=int(escalate),selected_budget=8 if escalate else 6,policy_pred=selected_pred,policy_correct=int(selected_pred==label),policy_latency_ms=t6+second_ms,policy_second_pass_ms=second_ms)
  for mode in modes:
   logits,ms,s,r=measured[mode];row.update({mode+'_pred':preds[mode],mode+'_correct':int(preds[mode]==label),mode+'_latency_ms':ms,mode+'_temporal_s':s,mode+'_spatial_r':r})
  if h27:
   ref=h27[i]
   if (str(ref['path'])!=path or int(ref['label'])!=label or int(ref['escalated'])!=int(escalate) or int(ref['selected_pred'])!=selected_pred or int(ref['s6_pred'])!=preds['S6']):raise RuntimeError(f'H27 policy/reference mismatch at index {i}')
  rows.append(row)
  if (i+1)%50==0 or i+1==len(loader):print(f'H28 validation: {i+1}/{len(loader)}',flush=True)
 n=len(rows);escalated=sum(r['escalated'] for r in rows)
 def metric(prefix):
  key='policy' if prefix=='policy' else prefix
  return dict(correct_n=sum(r[key+'_correct'] for r in rows),top1_pct=100*sum(r[key+'_correct'] for r in rows)/n,mean_forward_ms=statistics.mean(r[key+'_latency_ms'] for r in rows),median_forward_ms=statistics.median(r[key+'_latency_ms'] for r in rows),p95_forward_ms=sorted(r[key+'_latency_ms'] for r in rows)[int(.95*(n-1))])
 metrics={mode:metric(mode) for mode in (*modes,'policy')}
 paired={}
 for mode in modes:
  paired['policy_vs_'+mode]=dict(both_correct=sum(r['policy_correct'] and r[mode+'_correct'] for r in rows),policy_only_correct=sum(r['policy_correct'] and not r[mode+'_correct'] for r in rows),baseline_only_correct=sum(not r['policy_correct'] and r[mode+'_correct'] for r in rows),both_incorrect=sum(not r['policy_correct'] and not r[mode+'_correct'] for r in rows))
 summary=dict(experiment='Clean H28 matched validation forward latency and paired error audit',n=n,threshold=threshold,frozen_rule_sha256=sha(a.frozen_rule),manifest_sha256=digest,checkpoint_sha256=dict(h1=sha(a.dense_checkpoint),h13=sha(a.h13_checkpoint)),metrics=metrics,paired=paired,escalated_n=escalated,escalated_pct=100*escalated/n,mean_policy_full_model_passes=1+escalated/n,latency_protocol='Sequential synchronized forward only; rotating baseline order; policy S6 plus separate S8 second pass; batch=1; preprocessing/data transfer excluded',h27_reference_checked=bool(h27),official_test_evaluated=False,limitations=['One validation set, not independent test','No threshold retuning','No physical token deletion or FLOPs reduction','Timing varies with GPU environment; no speedup inferred from retention'])
 with (out/'per_video_validation.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print('H28 RESULT:',json.dumps(summary),flush=True)
 print('CLEAN H28 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
