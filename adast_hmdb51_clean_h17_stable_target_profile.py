#!/usr/bin/env python3
"""Clean H17: audit H16 per-view targets, no GPU, no model training."""
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--views-csv',default='runs/hmdb51_clean_h16_target_reliability_s1/view_targets.csv')
    p.add_argument('--protocol',default='runs/hmdb51_clean_h16_target_reliability_s1/protocol.json')
    p.add_argument('--h16-summary',default='runs/hmdb51_clean_h16_target_reliability_s1/summary.json')
    p.add_argument('--output',default='runs/hmdb51_clean_h17_stable_target_profile_s1')
    a=p.parse_args()
    src,pp,sp=map(Path,(a.views_csv,a.protocol,a.h16_summary))
    for path in (src,pp,sp):
        if not path.is_file(): raise FileNotFoundError(path)
    protocol=json.loads(pp.read_text(encoding='utf-8'))
    prior=json.loads(sp.read_text(encoding='utf-8'))
    nviews=int(protocol['args']['views'])
    selected=set(map(int,protocol['selected_indices']))
    if nviews<2 or len(selected)!=len(protocol['selected_indices']): raise ValueError('Invalid protocol views/selected_indices')
    groups=defaultdict(list)
    with src.open(newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        required={'index','path','label','original_target','view','target_s','correct_s8','loss_s6','loss_s7','loss_s8','correct_s6','correct_s7'}
        if not required.issubset(r.fieldnames or []):raise ValueError('Missing CSV columns: '+str(required-set(r.fieldnames or [])))
        for row in r:
            i=int(row['index']);v=int(row['view']);s=int(row['target_s']);o=int(row['original_target'])
            if i not in selected or not 1<=v<=nviews or s not in (6,7,8) or o not in (6,7,8):raise ValueError('Invalid row index/view/target')
            groups[i].append(row)
    if set(groups)!=selected:raise ValueError('CSV video indices do not match protocol selected_indices')
    output=[];counts=Counter();by_original=defaultdict(Counter)
    for i,rows in sorted(groups.items()):
        if len(rows)!=nviews or {int(r['view']) for r in rows}!=set(range(1,nviews+1)):raise ValueError(f'Incomplete/duplicate views for index {i}')
        if len({(r['path'],r['label'],r['original_target']) for r in rows})!=1:raise ValueError(f'Inconsistent video metadata {i}')
        rows.sort(key=lambda r:int(r['view']))
        original=int(rows[0]['original_target']);ts=[int(r['target_s']) for r in rows];freq=Counter(ts)
        unanimous=len(freq)==1
        stable_original=unanimous and ts[0]==original
        all_s8_correct=all(int(r['correct_s8'])==1 for r in rows)
        # S8 wrong on any view invalidates a claim of an always-safe budget for this video.
        strict_eligible=stable_original and all_s8_correct
        majority=max(freq.values())>nviews/2
        majority_target=next((s for s,c in freq.items() if c>nviews/2),None)
        record={'index':i,'path':rows[0]['path'],'label':int(rows[0]['label']),'original_target':original,
                'view_targets':'|'.join(map(str,ts)),'view_s6':freq[6],'view_s7':freq[7],'view_s8':freq[8],
                'unanimous':int(unanimous),'unanimous_target':ts[0] if unanimous else '',
                'stable_original':int(stable_original),'all_s8_correct':int(all_s8_correct),
                'strict_eligible':int(strict_eligible),'majority_target':majority_target if majority else '',
                'majority_matches_original':int(majority and majority_target==original),
                'any_s8_wrong':int(not all_s8_correct)}
        output.append(record)
        for key in ('unanimous','stable_original','strict_eligible','majority_matches_original','any_s8_wrong'):
            counts[key]+=record[key];by_original[original][key]+=record[key]
        by_original[original]['n']+=1
        if strict_eligible:counts[f'strict_s{original}']+=1
    if len(output)!=int(prior['selected_n']) or len(output)*nviews!=int(prior['view_total']):raise ValueError('H16 summary count mismatch')
    if counts['unanimous']!=int(prior['all_views_agree_n']) or counts['stable_original']!=int(prior['all_views_match_original_n']):raise ValueError('H16 summary agreement mismatch')
    if sum(r['all_s8_correct'] for r in output)!=int(prior['s8_correct_all_views_n']):raise ValueError('H16 summary S8 correctness mismatch')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'per_video_profile.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    with (out/'strict_stable_targets.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['index','path','label','target_s','view_targets'])
        w.writeheader()
        for r in output:
            if r['strict_eligible']:w.writerow({k:r[k] if k!='target_s' else r['original_target'] for k in w.fieldnames})
    summary={'experiment':'Clean H17 H16 sampled training-only stable-target audit','sample_n':len(output),
             'views_per_video':nviews,'source_csv_sha256':sha256(src),'source_protocol_sha256':sha256(pp),
             'counts':dict(sorted(counts.items())),'by_original_target':{str(k):dict(v) for k,v in sorted(by_original.items())},
             'strict_definition':'Original H9-A target equals every H16 view target AND S8 predicts correctly on every view',
             'cautions':['H16 sample balanced across original target classes, not representative of all 3213 training videos.',
                         'Stable on three sampled temporal views is not a guarantee for unseen views or official test.',
                         'S8 incorrect on any sampled view excludes strict eligibility; S8 correctness alone does not certify smaller-budget safety.',
                         'No validation/test data accessed; no GPU or training; shape-preserving gating unchanged.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('H17 RESULT: '+json.dumps(summary,ensure_ascii=False))
    print('CLEAN H17 COMPLETE: '+str((out/'summary.json').resolve()))
if __name__=='__main__':main()
