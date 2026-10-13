#!/usr/bin/env python3
"""H23 validation-only retrospective oracle audit. Labels are used for selection: NOT deployable."""
import argparse, csv, hashlib, json, math
from collections import Counter
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=357 or len({int(r['index']) for r in rows})!=357 or {int(r['index']) for r in rows}!=set(range(357)):
        raise RuntimeError(f'{path}: expected exactly 357 unique indices 0..356')
    return {int(r['index']):r for r in rows}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.input_dir);out=Path(a.output)
    if out.exists() and any(out.iterdir()):p.error('Output must be a new/empty directory')
    names=['h13_s6','h13_s7','h13_s8','h13_adaptive']
    paths={n:root/(n+'_per_video.csv') for n in names}
    summary_path=root/'summary.json';protocol_path=root/'protocol.json'
    for path in list(paths.values())+[summary_path]:
        if not path.is_file():raise FileNotFoundError(path)
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    protocol=json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.is_file() else summary.get('protocol',{})
    if protocol.get('val_n')!=357 or protocol.get('official_test_n_not_evaluated')!=1530 or protocol.get('physical_token_deletion') is not False:
        raise RuntimeError('Unexpected H21 protocol metadata')
    result=summary.get('results',summary)
    data={name:load(path) for name,path in paths.items()}
    for name in names:
        rows=data[name]
        for i in range(357):
            r=rows[i]
            if int(r['correct']) != int(int(r['pred'])==int(r['label'])):raise RuntimeError(f'{name} row {i}: incorrect correctness field')
            if name!='h13_adaptive' and int(r['temporal_s'])!=int(name[-1]):raise RuntimeError(f'{name} row {i}: forced budget mismatch')
            if name=='h13_adaptive' and int(r['temporal_s']) not in (6,7,8):raise RuntimeError('Invalid adaptive budget')
            base=data[names[0]][i]
            if (r['path'],r['label'])!=(base['path'],base['label']):raise RuntimeError(f'Video/label mismatch at index {i}')
        expected=result.get(name,{})
        if expected:
            actual=100*sum(int(r['correct']) for r in rows.values())/357
            if not math.isclose(actual,float(expected['top1']),abs_tol=1e-8):raise RuntimeError(f'{name}: H21 summary Top-1 mismatch')
    audit=[];counts=Counter();oracle_correct=0;oracle_ret=0;adaptive_correct=0;adaptive_ret=0
    for i in range(357):
        fixed={s:data[f'h13_s{s}'][i] for s in (6,7,8)};ad=data['h13_adaptive'][i]
        feasible=[s for s in (6,7,8) if int(fixed[s]['correct'])]
        # For videos where no fixed budget is correct, choose S6 to minimize retention.
        # This is an OPTIMISTIC accuracy/retention upper bound, not a safe fallback policy.
        chosen=min(feasible) if feasible else 6
        r=fixed[chosen];counts[f'oracle_s{chosen}']+=1
        if not feasible:counts['no_fixed_budget_correct']+=1
        if len(feasible)==3:counts['all_fixed_budgets_correct']+=1
        if feasible and 6 not in feasible:counts['requires_s7_or_s8_for_correctness']+=1
        if feasible and 6 not in feasible and 7 not in feasible:counts['requires_s8_for_correctness']+=1
        oc=int(r['correct']);ac=int(ad['correct']);oracle_correct+=oc;adaptive_correct+=ac
        oracle_ret+=float(r['effective_retention']);adaptive_ret+=float(ad['effective_retention'])
        audit.append(dict(index=i,path=r['path'],label=r['label'],correct_s6=int(fixed[6]['correct']),correct_s7=int(fixed[7]['correct']),correct_s8=int(fixed[8]['correct']),feasible_budgets=';'.join(map(str,feasible)),oracle_budget=chosen,oracle_correct=oc,oracle_retention=r['effective_retention'],adaptive_budget=ad['temporal_s'],adaptive_correct=ac,adaptive_retention=ad['effective_retention'],oracle_only_correct=int(oc and not ac),adaptive_only_correct=int(ac and not oc)))
    ret=lambda name:sum(float(row['effective_retention']) for row in data[name].values())/357
    top=lambda name:100*sum(int(row['correct']) for row in data[name].values())/357
    output=dict(experiment='Clean H23 retrospective validation-only minimum-correct-budget oracle',n=357,counts=dict(sorted(counts.items())),fixed={n:dict(top1=top(n),mean_effective_retention=ret(n)) for n in names[:3]},adaptive=dict(top1=top('h13_adaptive'),mean_effective_retention=ret('h13_adaptive')),oracle=dict(top1=100*oracle_correct/357,mean_effective_retention=oracle_ret/357,correct_n=oracle_correct),paired_oracle_vs_adaptive=dict(oracle_only_correct=sum(r['oracle_only_correct'] for r in audit),adaptive_only_correct=sum(r['adaptive_only_correct'] for r in audit)),source_sha256={p.name:digest(p) for p in list(paths.values())+[summary_path]+([protocol_path] if protocol_path.is_file() else [])},limitations=['Oracle selects budgets using ground-truth correctness on the SAME validation videos; not an inference-time controller, unbiased model-selection estimate, or deployable result.','For videos where all three fixed budgets fail, chooses S6 for an optimistic minimum-retention envelope; not a safety fallback.','Only three discrete forced budgets and one deterministic view are covered.','Effective retention describes shape-preserving gating, not actual FLOPs, latency, or physical token deletion.','Official test data not accessed.'])
    out.mkdir(parents=True,exist_ok=True)
    with (out/'oracle_per_video.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    (out/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('H23 RESULT:',json.dumps({k:output[k] for k in ('n','counts','fixed','adaptive','oracle','paired_oracle_vs_adaptive')}),flush=True)
    print('CLEAN H23 COMPLETE:',out/'summary.json',flush=True)
if __name__=='__main__':main()
