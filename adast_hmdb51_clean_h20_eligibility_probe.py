#!/usr/bin/env python3
"""Clean H20: training-only, exploratory H19 eligibility learnability audit.
Requires numpy and scikit-learn. Does not access validation/test or train AdaST.
"""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CATS = {'eligible_s6_all_views': 6, 'eligible_s7_all_views': 7,
        'eligible_s8_all_views_only': 8}

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def metrics(y,p):
    return {'accuracy_percent':float(100*accuracy_score(y,p)),
            'balanced_accuracy_percent':float(100*balanced_accuracy_score(y,p)),
            'macro_f1_percent':float(100*f1_score(y,p,labels=[6,7,8],average='macro',zero_division=0)),
            'recall_percent':{str(k):float(100*recall_score(y,p,labels=[k],average='macro',zero_division=0)) for k in (6,7,8)},
            'confusion_true_rows_pred_columns':confusion_matrix(y,p,labels=[6,7,8]).tolist()}

def aggregate(items):
    keys=('accuracy_percent','balanced_accuracy_percent','macro_f1_percent')
    return {k:{'mean':float(np.mean([r[k] for r in items])),
               'std_across_folds':float(np.std([r[k] for r in items]))} for k in keys}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eligibility-csv',required=True)
    ap.add_argument('--h19-summary',required=True)
    ap.add_argument('--h15-features',required=True)
    ap.add_argument('--h15-protocol',required=True)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    paths=[Path(a.eligibility_csv),Path(a.h19_summary),Path(a.h15_features),Path(a.h15_protocol)]
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    h19=json.loads(paths[1].read_text(encoding='utf-8'))
    h15=json.loads(paths[3].read_text(encoding='utf-8'))
    if h19.get('sample_n')!=300 or h19.get('views_per_video')!=3:
        raise ValueError('Unexpected H19 audit metadata')
    if h15.get('feature_source','').find('exact 1536-D')<0 or h15.get('train_n')!=3213:
        raise ValueError('Unexpected H15 feature provenance')
    with np.load(paths[2],allow_pickle=False) as z:
        if not {'features','targets','fit_indices','holdout_indices'}<=set(z.files):
            raise ValueError('Missing H15 feature cache arrays')
        features=z['features'].copy(); old_targets=z['targets'].copy()
    if features.shape!=(3213,1536) or old_targets.shape!=(3213,) or not np.isfinite(features).all():
        raise ValueError('Unexpected feature dimensions or nonfinite features')
    rows=list(csv.DictReader(paths[0].open(newline='',encoding='utf-8-sig')))
    if len(rows)!=300 or len({int(r['index']) for r in rows})!=300:
        raise ValueError('Expected 300 distinct sampled train indices')
    indices=[]; targets=[]; excluded=[]
    for r in rows:
        i=int(r['index']); cat=r['category']; orig=int(r['original_target'])
        if not 0<=i<3213 or int(old_targets[i])!=orig:
            raise ValueError(f'H15/H19 index or original target mismatch: {i}')
        if cat=='unresolved_s8_error':
            if r['s8_correct_all_views']!='0' or r['observed_robust_budget'] not in ('','unresolved'):
                raise ValueError(f'Invalid unresolved record {i}')
            excluded.append(i);continue
        if cat not in CATS or int(r['observed_robust_budget'])!=CATS[cat]:
            raise ValueError(f'Invalid eligibility category {i}')
        if r['s8_correct_all_views']!='1':raise ValueError(f'S8 incorrect in eligible record {i}')
        indices.append(i);targets.append(CATS[cat])
    counts=Counter(targets)
    expected={6:h19['category_counts']['eligible_s6_all_views'],
              7:h19['category_counts']['eligible_s7_all_views'],
              8:h19['category_counts']['eligible_s8_all_views_only']}
    if dict(counts)!=expected or len(excluded)!=h19['category_counts']['unresolved_s8_error']:
        raise ValueError('H19 category totals mismatch')
    if min(counts.values())<a.folds or a.folds<2 or a.repeats<1:
        raise ValueError('Invalid fold/repeat count')
    X=features[np.asarray(indices)];y=np.asarray(targets,dtype=np.int64)
    cv=RepeatedStratifiedKFold(n_splits=a.folds,n_repeats=a.repeats,random_state=a.seed)
    results=[]; baselines=[];fold_rows=[]
    for n,(tr,te) in enumerate(cv.split(X,y),1):
        # Scaler and probe are fitted exclusively within the current training fold.
        model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=a.seed))
        model.fit(X[tr],y[tr]);p=model.predict(X[te]);b=np.full_like(y[te],8)
        r=metrics(y[te],p);br=metrics(y[te],b)
        results.append(r);baselines.append(br)
        fold_rows.append({'fold':n,'fit_n':len(tr),'heldout_n':len(te),'probe':r,'always_s8':br})
        print(f'H20 FOLD: {n}/{a.folds*a.repeats} | probe balanced accuracy={r["balanced_accuracy_percent"]:.2f}%',flush=True)
    result={'experiment':'Clean H20 training-only H19 three-view eligibility probe',
            'eligible_n':len(y),'excluded_unresolved_n':len(excluded),
            'target_counts':{str(k):int(counts[k]) for k in (6,7,8)},
            'feature_dim':int(X.shape[1]),'folds':a.folds,'repeats':a.repeats,'seed':a.seed,
            'probe':aggregate(results),'always_s8':aggregate(baselines),
            'probe_mean_recall_percent':{str(k):float(np.mean([r['recall_percent'][str(k)] for r in results])) for k in (6,7,8)},
            'source_sha256':{p.name:digest(p) for p in paths},
            'limitations':['Training-only exploratory probe; no validation or official test accessed.',
                           'H19 300-video sample stratified by original H9-A targets, not population representative.',
                           'Seven S8-error videos excluded, so probe is not an end-to-end safety or accuracy estimate.',
                           'H19 eligibility uses ground-truth labels and forced-budget losses, unavailable at inference.',
                           'Features are deterministic single-view H15 inputs; targets come from three H16 temporal views.',
                           'Fold standard deviations are descriptive; repeated folds are not independent replicates.',
                           'Shape-preserving gating is not physical token deletion or demonstrated speedup.']}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (out/'folds.json').write_text(json.dumps(fold_rows,indent=2)+'\n',encoding='utf-8')
    print('H20 RESULT: '+json.dumps(result),flush=True)
    print('CLEAN H20 COMPLETE: '+str((out/'summary.json').resolve()),flush=True)
if __name__=='__main__':main()
