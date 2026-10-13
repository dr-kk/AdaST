#!/usr/bin/env python3
"""Clean H25: training-only retrospective S6 early-exit signal audit. CPU only.
No validation/test access; no trained controller; no deployable speedup claim.
"""
import argparse, csv, json, pathlib
import numpy as np

def softmax(x):
    z=x.astype(np.float64); z-=z.max(axis=1,keepdims=True)
    p=np.exp(z); return p/p.sum(axis=1,keepdims=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,help='H24 train_fixed_logits.npz')
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with np.load(args.input,allow_pickle=False) as d:
        required=['indices','labels','paths','logits_s6','logits_s7','logits_s8']
        if any(k not in d.files for k in required):raise ValueError('Missing NPZ keys: '+str(set(required)-set(d.files)))
        ids=d['indices'].copy();labels=d['labels'].copy();paths=d['paths'].copy()
        logits={b:d[f'logits_s{b}'].copy() for b in (6,7,8)}
    n=len(labels)
    if n!=3213 or len(set(ids.tolist()))!=n or any(x.shape!=(n,51) for x in logits.values()):
        raise ValueError('Unexpected train size, duplicate indices, or logits shapes')
    if not np.all(np.isfinite(np.concatenate(list(logits.values()),axis=1))):raise ValueError('Nonfinite logits')
    p={b:softmax(logits[b]) for b in (6,7,8)}
    pred={b:p[b].argmax(axis=1) for b in (6,7,8)}
    correct={b:pred[b]==labels for b in (6,7,8)}
    sorted_p=np.sort(p[6],axis=1)
    signals={'confidence':sorted_p[:,-1], 'margin':sorted_p[:,-1]-sorted_p[:,-2],
             'entropy':-(p[6]*np.log(np.clip(p[6],1e-300,1))).sum(axis=1)}
    # A rescue requires S6 incorrect and at least one higher-budget prediction correct.
    rescue=(~correct[6]) & (correct[7]|correct[8])
    # S6-correct videos that would be harmed by forcing S8.
    harmed_s8=correct[6]&(~correct[8])
    # Report conditional results for a simple two-pass policy: S6 then S8 on escalation.
    # Retention is a proxy for selected budget only, NOT cumulative compute or speedup.
    r6=6/8*.5987394957983193; r8=.5987394957983193
    rows=[]
    for name,values in signals.items():
        direction='low' if name!='entropy' else 'high'
        for q in (1,2,5,10,15,20,25,30,40,50,60,75,90,100):
            if q==100: escalate=np.ones(n,dtype=bool)
            else:
                # Deterministic rank-based escalation; ties broken by input index.
                k=int(np.floor(n*q/100));order=np.argsort(values,kind='stable')
                if direction=='high':order=order[::-1]
                escalate=np.zeros(n,dtype=bool);escalate[order[:k]]=True
            chosen_correct=np.where(escalate,correct[8],correct[6])
            rows.append({'signal':name,'direction':direction,'target_escalation_pct':q,
                'escalated_n':int(escalate.sum()),'escalated_pct':float(100*escalate.mean()),
                'top1_pct':float(100*chosen_correct.mean()),
                'top1_delta_vs_s6_pp':float(100*(chosen_correct.mean()-correct[6].mean())),
                'top1_delta_vs_s8_pp':float(100*(chosen_correct.mean()-correct[8].mean())),
                'rescue_captured_n':int((escalate&rescue).sum()),
                'rescue_total_n':int(rescue.sum()),
                'rescue_recall_pct':float(100*(escalate&rescue).sum()/max(1,rescue.sum())),
                's6_correct_escalated_n':int((escalate&correct[6]).sum()),
                's6_correct_harmed_by_s8_n':int((escalate&harmed_s8).sum()),
                'selected_budget_retention_proxy':float(np.where(escalate,r8,r6).mean())})
    with (out/'signal_sweep_train_only.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'per_video_train_only.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['index','path','label','pred_s6','pred_s7','pred_s8','correct_s6','correct_s7','correct_s8','rescuable_by_s7_or_s8','confidence_s6','margin_s6','entropy_s6']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(n):w.writerow(dict(index=int(ids[i]),path=str(paths[i]),label=int(labels[i]),
          pred_s6=int(pred[6][i]),pred_s7=int(pred[7][i]),pred_s8=int(pred[8][i]),
          correct_s6=int(correct[6][i]),correct_s7=int(correct[7][i]),correct_s8=int(correct[8][i]),
          rescuable_by_s7_or_s8=int(rescue[i]),confidence_s6=float(signals['confidence'][i]),
          margin_s6=float(signals['margin'][i]),entropy_s6=float(signals['entropy'][i])))
    summary={'experiment':'Clean H25 training-only retrospective S6 signal audit',
      'n':n,'top1_pct':{str(b):float(100*correct[b].mean()) for b in (6,7,8)},
      's6_incorrect_n':int((~correct[6]).sum()),'s6_incorrect_rescuable_by_s7_or_s8_n':int(rescue.sum()),
      's6_incorrect_rescuable_by_s8_n':int((~correct[6]&correct[8]).sum()),
      's6_correct_harmed_by_s8_n':int(harmed_s8.sum()),
      's6_s7_disagreement_n':int((pred[6]!=pred[7]).sum()),
      's6_s8_disagreement_n':int((pred[6]!=pred[8]).sum()),
      'sweep':'S6-only signal rank sweep, retrospective TRAIN labels for evaluation; S6->S8 conditional second pass',
      'outputs':['signal_sweep_train_only.csv','per_video_train_only.csv'],
      'limitations':['Train-only in-sample audit, not validation/test or generalization evidence',
      'Percentile ranking over all train videos is an offline diagnostic, not a deployable per-video threshold',
      'S6 then S8 requires two passes; selected-budget retention proxy excludes S6 first-pass cost',
      'Shape-preserving gating; retention is not physical pruning, FLOPs, latency, or speedup',
      'Do not use labels or S7/S8 predictions as inference-time decision inputs']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('H25 RESULT:',json.dumps(summary));print('CLEAN H25 COMPLETE:',out.resolve())
if __name__=='__main__':main()
