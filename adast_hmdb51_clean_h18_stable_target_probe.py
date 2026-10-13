#!/usr/bin/env python
"""H18: training-only exploratory stable-target learnability using cached H15 exact controller features.
No GPU, model training, validation or official test. Probe fit is restricted to CV training folds.
"""
import argparse,csv,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,recall_score

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stable-csv',required=True)
 p.add_argument('--h17-summary',required=True)
 p.add_argument('--h15-features',required=True,help='H15 train_features_and_split.npz')
 p.add_argument('--h15-protocol',required=True)
 p.add_argument('--profile',required=True,help='H9-A train_budget_profile.csv for index/path/label cross-check')
 p.add_argument('--output',required=True)
 p.add_argument('--folds',type=int,default=5)
 p.add_argument('--repeats',type=int,default=3)
 p.add_argument('--seed',type=int,default=42)
 a=p.parse_args()
 if a.folds<2 or a.repeats<1:p.error('folds >=2 and repeats >=1 required')
 out=Path(a.output).resolve()
 if out.exists() and any(out.iterdir()):p.error('Output directory must be empty; choose a new path')
 h17=json.loads(Path(a.h17_summary).read_text(encoding='utf-8'))
 h15=json.loads(Path(a.h15_protocol).read_text(encoding='utf-8'))
 if h17.get('counts',{}).get('strict_eligible') is None:raise RuntimeError('Not an H17 summary')
 if h15.get('profile_sha256')!=sha(a.profile):raise RuntimeError('H15 cached features belong to a different H9-A profile')
 with open(a.profile,newline='',encoding='utf-8') as f:profile={int(r['index']):r for r in csv.DictReader(f)}
 if len(profile)!=h15['train_n'] or set(profile)!=set(range(h15['train_n'])):raise RuntimeError('Incomplete H9-A profile')
 with open(a.stable_csv,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 if len(rows)!=h17['counts']['strict_eligible']:raise RuntimeError('H17 strict count mismatch')
 if len({int(r['index']) for r in rows})!=len(rows):raise RuntimeError('Duplicate stable indices')
 indices=[];targets=[]
 for r in rows:
  i=int(r['index']);t=int(r['target_s'])
  if i not in profile or t not in (6,7,8) or r['path']!=profile[i]['path'] or int(r['label'])!=int(profile[i]['label']) or t!=int(profile[i]['target_s']) or r['view_targets']!=f'{t}|{t}|{t}':raise RuntimeError(f'Stable/profile mismatch at {i}')
  indices.append(i);targets.append(t)
 with np.load(a.h15_features,allow_pickle=False) as z:
  Xall=z['features'];yall=z['targets']
 if Xall.shape!=(h15['train_n'],1536) or yall.shape!=(h15['train_n'],):raise RuntimeError('H15 cache shape mismatch')
 indices=np.asarray(indices,dtype=int);y=np.asarray(targets,dtype=int)
 if not np.array_equal(yall[indices],y):raise RuntimeError('H15 feature cache targets mismatch')
 X=np.asarray(Xall[indices],dtype=np.float32)
 if not np.isfinite(X).all():raise RuntimeError('Nonfinite features')
 counts=Counter(y.tolist())
 if min(counts.values())<a.folds:raise RuntimeError('Not enough samples per class for requested folds')
 splitter=RepeatedStratifiedKFold(n_splits=a.folds,n_repeats=a.repeats,random_state=a.seed)
 results=[];pred_rows=[]
 for k,(tr,te) in enumerate(splitter.split(X,y),1):
  model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=a.seed))
  model.fit(X[tr],y[tr]);pred=model.predict(X[te]);base=np.full(len(te),6,dtype=int)
  def score(p):return {'accuracy_percent':100*accuracy_score(y[te],p),'balanced_accuracy_percent':100*balanced_accuracy_score(y[te],p),'macro_f1_percent':100*f1_score(y[te],p,labels=[6,7,8],average='macro',zero_division=0),'recall_s6_s7_s8':recall_score(y[te],p,labels=[6,7,8],average=None,zero_division=0).tolist(),'confusion_rows_true_cols_pred':confusion_matrix(y[te],p,labels=[6,7,8]).tolist()}
  results.append({'fold':k,'fit_n':len(tr),'holdout_n':len(te),'probe':score(pred),'always_s6':score(base)})
  for j,prediction in zip(te,pred):pred_rows.append({'fold':k,'index':int(indices[j]),'target_s':int(y[j]),'predicted_s':int(prediction)})
  print(f'H18 FOLD: {k}/{a.folds*a.repeats} | probe balanced accuracy={results[-1]["probe"]["balanced_accuracy_percent"]:.2f}%',flush=True)
 def aggregate(name):
  return {m:{'mean':float(np.mean([r[name][m] for r in results])),'std_across_folds':float(np.std([r[name][m] for r in results],ddof=1))} for m in ('accuracy_percent','balanced_accuracy_percent','macro_f1_percent')}
 summary={'experiment':'H18 exploratory stable-target learnability, training-only','stable_n':len(y),'counts':{str(s):int(counts[s]) for s in (6,7,8)},'feature_dim':1536,'folds':a.folds,'repeats':a.repeats,'total_folds':len(results),'probe':aggregate('probe'),'always_s6':aggregate('always_s6'),'per_fold':results,'provenance':{'stable_csv_sha256':sha(a.stable_csv),'h17_summary_sha256':sha(a.h17_summary),'h15_features_sha256':sha(a.h15_features),'h15_protocol_sha256':sha(a.h15_protocol),'h9_profile_sha256':sha(a.profile)},'limitations':['Stable subset selected using three H16 temporal views; only H15 deterministic-view features used.','Fold metrics are correlated; std across folds is not a confidence interval.','Only 19 S7 and 28 S8 examples; results are exploratory.','H16 selected equal numbers from each original target, so this subset is not population-representative.','Probe predicts pseudo-budget targets, not action labels, runtime gains or deployment performance.','No validation or official test evaluated; shape-preserving gating unchanged.']}
 out.mkdir(parents=True,exist_ok=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 with (out/'out_of_fold_predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['fold','index','target_s','predicted_s']);w.writeheader();w.writerows(pred_rows)
 print('H18 RESULT:',json.dumps({k:summary[k] for k in ('stable_n','counts','feature_dim','folds','repeats','probe','always_s6')}),flush=True)
 print('CLEAN H18 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
