#!/usr/bin/env python3
"""H26: train-only S6->S8 fixed threshold selection; no validation/test access."""
import argparse, json, pathlib
import numpy as np
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--max-escalation-pct',type=float,default=10.0);a=p.parse_args()
 d=pd.read_csv(a.input); required=['index','correct_s6','correct_s8','confidence_s6','margin_s6','entropy_s6']
 assert all(x in d for x in required),f'Missing: {set(required)-set(d)}'
 assert len(d)==3213 and d['index'].is_unique and not d[required].isna().any().any()
 for c in ['correct_s6','correct_s8']: assert set(d[c].unique()).issubset({0,1})
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
 n=len(d);s6=d.correct_s6.to_numpy(dtype=int);s8=d.correct_s8.to_numpy(dtype=int)
 rows=[]; thresholds={}
 for name,direction in [('confidence_s6','low'),('margin_s6','low'),('entropy_s6','high')]:
  v=d[name].to_numpy(dtype=float);assert np.isfinite(v).all()
  for budget in [1,2,5,10,15,20,30,40,50]:
   # A numerical threshold determined from train signals only, never per-video train rank at inference.
   t=float(np.quantile(v,budget/100 if direction=='low' else 1-budget/100))
   escalate=v<=t if direction=='low' else v>=t
   rate=100*escalate.mean();pred=np.where(escalate,s8,s6)
   rescued=int(((s6==0)&(s8==1)&escalate).sum()); harmed=int(((s6==1)&(s8==0)&escalate).sum())
   rows.append(dict(signal=name,direction=direction,requested_train_quantile_pct=budget,threshold=t,comparison='<=' if direction=='low' else '>=',escalated_n=int(escalate.sum()),escalated_pct=rate,train_top1_pct=100*pred.mean(),train_delta_vs_s6_pp=100*(pred.mean()-s6.mean()),train_delta_vs_s8_pp=100*(pred.mean()-s8.mean()),rescue_captured_n=rescued,s6_correct_harmed_n=harmed,mean_full_model_passes=1+escalate.mean(),note='S6 then S8: second full pass on escalated videos; not FLOPs/latency/speedup'))
 r=pd.DataFrame(rows);r.to_csv(out/'threshold_candidates_train_only.csv',index=False)
 eligible=r[r.escalated_pct<=a.max_escalation_pct+1e-9]
 if eligible.empty: raise RuntimeError('No threshold meets requested escalation cap')
 # Fixed train-only choice, deterministic tie break, explicitly exploratory and in-sample.
 best=eligible.sort_values(['train_top1_pct','escalated_pct','signal'],ascending=[False,True,True]).iloc[0].to_dict()
 frozen=dict(experiment='Clean H26 train-only fixed numerical threshold selection',n=n,signal=best['signal'],direction=best['direction'],threshold=float(best['threshold']),comparison=best['comparison'],train_escalation_cap_pct=a.max_escalation_pct,train_escalated_pct=float(best['escalated_pct']),train_top1_pct=float(best['train_top1_pct']),train_s6_top1_pct=float(100*s6.mean()),train_s8_top1_pct=float(100*s8.mean()),mean_full_model_passes=float(best['mean_full_model_passes']),inference_policy='Run S6; if S6 signal meets fixed threshold, run S8 as a second full pass and use S8 output; otherwise use S6 output.',limitations=['Selected and evaluated on same train videos; optimistic in-sample result, not validation evidence','Threshold is a fixed numerical value, not a per-batch percentile/rank','True labels used for train-only selection and audit, never inference-time input','S6 then S8 requires two full forward passes on escalated videos','Shape-preserving gating: no measured FLOPs/latency/speedup','No validation or official test accessed'])
 (out/'frozen_rule_train_only.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
 print('H26 RESULT:',json.dumps(frozen));print('H26 COMPLETE:',out)
if __name__=='__main__':main()
