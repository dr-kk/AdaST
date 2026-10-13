#!/usr/bin/env python3
"""Clean H22: CPU-only paired audit of H21 validation CSVs; no video decoding or training."""
import argparse,csv,hashlib,json,math
from pathlib import Path
from collections import Counter

MODES=('h2_s6','h2_s7','h2_s8','h2_adaptive','h13_s6','h13_s7','h13_s8','h13_adaptive')
FIELDS=('index','path','label','pred','correct','top5','nll','spatial_r','temporal_s','k_per_frame','effective_retention')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:
  reader=csv.DictReader(f)
  if not set(FIELDS).issubset(reader.fieldnames or []):raise ValueError(f'{p}: required CSV columns missing: {set(FIELDS)-set(reader.fieldnames or [])}')
  rows=list(reader)
 if len(rows)!=357:raise ValueError(f'{p}: expected 357 rows, got {len(rows)}')
 out={}
 for r in rows:
  i=int(r['index'])
  if i in out or not 0<=i<357:raise ValueError(f'{p}: duplicate/invalid index {i}')
  for k in ('label','pred','correct','top5','temporal_s','k_per_frame'):r[k]=int(r[k])
  for k in ('nll','spatial_r','effective_retention'):r[k]=float(r[k])
  if r['correct']!=int(r['pred']==r['label']) or r['correct'] not in (0,1) or r['top5'] not in (0,1) or r['top5']<r['correct']:raise ValueError(f'{p}: invalid prediction/accuracy index {i}')
  if not 0<r['spatial_r']<=1 or not 0<r['effective_retention']<=1 or r['temporal_s'] not in (6,7,8):raise ValueError(f'{p}: invalid retention/budget index {i}')
  if not math.isclose(r['effective_retention'],r['spatial_r']*r['temporal_s']/8,abs_tol=1e-6):raise ValueError(f'{p}: inconsistent effective retention index {i}')
  out[i]=r
 if set(out)!=set(range(357)):raise ValueError(f'{p}: incomplete index set')
 return out

def pair(a,b):
 both=sum(a[i]['correct'] and b[i]['correct'] for i in a)
 aonly=sum(a[i]['correct'] and not b[i]['correct'] for i in a)
 bonly=sum(b[i]['correct'] and not a[i]['correct'] for i in a)
 neither=357-both-aonly-bonly
 return dict(both_correct=both,adaptive_only_correct=aonly,fixed_only_correct=bonly,both_incorrect=neither,net_correct_delta=aonly-bonly,top1_delta_pp=100*(aonly-bonly)/357,discordant_n=aonly+bonly,adaptive_mean_retention=sum(r['effective_retention'] for r in a.values())/357,fixed_mean_retention=sum(r['effective_retention'] for r in b.values())/357)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input-dir',required=True,help='H21 output directory containing all eight *_per_video.csv files and summary.json')
 p.add_argument('--output',required=True,help='Fresh output directory')
 a=p.parse_args();root=Path(a.input_dir);out=Path(a.output)
 if not root.is_dir():p.error('Input directory not found')
 if out.exists() and any(out.iterdir()):p.error('Output directory must be absent or empty')
 summary_path=root/'summary.json';summary=json.loads(summary_path.read_text(encoding='utf-8'))
 if summary.get('official_test_evaluated') is not False or summary.get('protocol',{}).get('val_n')!=357 or summary.get('protocol',{}).get('physical_token_deletion') is not False:raise RuntimeError('H21 summary/protocol validation failed')
 rows={m:read(root/(m+'_per_video.csv')) for m in MODES}
 ref=rows['h13_adaptive'];identity=[(ref[i]['path'],ref[i]['label']) for i in range(357)]
 for m,rs in rows.items():
  if [(rs[i]['path'],rs[i]['label']) for i in range(357)]!=identity:raise RuntimeError(f'{m}: video order/path/label differs')
  for i,r in rs.items():
   if m.endswith(('_s6','_s7','_s8')) and r['temporal_s']!=int(m[-1]):raise RuntimeError(f'{m}: forced budget mismatch index {i}')
  calc=100*sum(r['correct'] for r in rs.values())/357
  if not math.isclose(calc,summary['results'][m]['top1'],abs_tol=1e-7):raise RuntimeError(f'{m}: top1 disagrees with H21 summary')
  ret=sum(r['effective_retention'] for r in rs.values())/357
  if not math.isclose(ret,summary['results'][m]['mean_effective_retention'],abs_tol=1e-7):raise RuntimeError(f'{m}: retention disagrees with H21 summary')
 for s in (6,7,8):
  for i in range(357):
   x,y=rows[f'h2_s{s}'][i],rows[f'h13_s{s}'][i]
   if x['pred']!=y['pred'] or x['correct']!=y['correct'] or not math.isclose(x['effective_retention'],y['effective_retention'],abs_tol=1e-7):raise RuntimeError(f'H2/H13 fixed S{s} mismatch at index {i}')
 pairs={f'h13_adaptive_vs_s{s}':pair(ref,rows[f'h13_s{s}']) for s in (6,7,8)}
 by_budget={}
 for s in (6,7,8):
  ids=[i for i in range(357) if ref[i]['temporal_s']==s]
  by_budget[str(s)]=dict(n=len(ids),adaptive_correct=sum(ref[i]['correct'] for i in ids),fixed_s7_correct=sum(rows['h13_s7'][i]['correct'] for i in ids),fixed_s8_correct=sum(rows['h13_s8'][i]['correct'] for i in ids),adaptive_only_vs_s7=sum(ref[i]['correct'] and not rows['h13_s7'][i]['correct'] for i in ids),s7_only_vs_adaptive=sum(rows['h13_s7'][i]['correct'] and not ref[i]['correct'] for i in ids),adaptive_only_vs_s8=sum(ref[i]['correct'] and not rows['h13_s8'][i]['correct'] for i in ids),s8_only_vs_adaptive=sum(rows['h13_s8'][i]['correct'] and not ref[i]['correct'] for i in ids))
 details=[]
 for i in range(357):
  r=ref[i];z=dict(index=i,path=r['path'],label=r['label'],selected_s=r['temporal_s'],adaptive_pred=r['pred'],adaptive_correct=r['correct'],adaptive_retention=r['effective_retention'])
  for s in (6,7,8):
   f=rows[f'h13_s{s}'][i];z[f's{s}_pred']=f['pred'];z[f's{s}_correct']=f['correct'];z[f's{s}_retention']=f['effective_retention'];z[f'correct_delta_vs_s{s}']=r['correct']-f['correct']
  details.append(z)
 out.mkdir(parents=True,exist_ok=True)
 with (out/'paired_per_video.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
 result=dict(experiment='Clean H22 paired H21 validation prediction audit',validation_n=357,paired=pairs,by_selected_budget=by_budget,mode_top1={m:summary['results'][m]['top1'] for m in MODES},mode_mean_effective_retention={m:summary['results'][m]['mean_effective_retention'] for m in MODES},source_sha256={p.name:digest(p) for p in [summary_path]+[root/(m+'_per_video.csv') for m in MODES]},limitations=['Validation-only retrospective paired comparison; same validation set previously used for checkpoint selection.','No official test evaluation or independent statistical replication.','Effective retention is a gating proxy, not physical token deletion or measured speedup.','Per-video outcomes cannot establish why the controller selected a budget.'])
 (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('H22 RESULT:',json.dumps(dict(paired=pairs,by_selected_budget=by_budget)),flush=True)
 print('CLEAN H22 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
