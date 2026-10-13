#!/usr/bin/env python3
"""H34: read-only provenance audit. Never rewrites a manifest or checkpoint."""
import argparse, hashlib, json, pathlib, sys

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def walk(root):
    for p in root.rglob('*'):
        if p.is_file() and (('manifest' in p.name.lower() and p.suffix.lower() in ('.json','.txt','.csv')) or p.name.lower()=='split_manifest.json'):
            yield p

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',default='runs')
    ap.add_argument('--dense-checkpoint',default=r'runs\hmdb51_clean_h1_dense_s1\best_val.pt')
    ap.add_argument('--h13-checkpoint',default=r'runs\hmdb51_clean_h13_balanced_sampling_s1\best_val.pt')
    ap.add_argument('--manifest',default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    ap.add_argument('--output',default=r'runs\hmdb51_clean_h34_manifest_audit_s1')
    ap.add_argument('--expected-hash',default='8b45fc6c63ea8602becd674f59f1f8892174ce352c46fe45a7745423f8405bec')
    args=ap.parse_args()
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    report={'experiment':'H34 read-only manifest provenance audit','expected_hash':args.expected_hash,'paths':{},'manifest_candidates':[],'status':'STOP_REVIEW'}
    for name,raw in [('dense',args.dense_checkpoint),('h13',args.h13_checkpoint),('current_manifest',args.manifest)]:
        p=pathlib.Path(raw)
        report['paths'][name]={'path':str(p),'exists':p.is_file(),'sha256':digest(p) if p.is_file() else None}
    roots=[pathlib.Path(args.runs)]
    for folder in ('backup','backups','archive','archives'):
        p=pathlib.Path(folder)
        if p.is_dir():roots.append(p)
    seen=set()
    for root in roots:
        if not root.is_dir():continue
        for p in walk(root):
            try:
                resolved=str(p.resolve())
                if resolved in seen:continue
                seen.add(resolved)
                sha=digest(p)
                item={'path':str(p),'sha256':sha,'matches_expected':sha==args.expected_hash,'size_bytes':p.stat().st_size}
                if p.suffix.lower()=='.json':
                    try:
                        obj=json.loads(p.read_text(encoding='utf-8-sig'))
                        item['json_type']=type(obj).__name__
                        if isinstance(obj,dict):item['top_level_keys']=list(obj)[:35]
                    except (UnicodeError,ValueError) as exc:item['json_parse_error']=str(exc)[:180]
                report['manifest_candidates'].append(item)
            except OSError as exc:report['manifest_candidates'].append({'path':str(p),'error':str(exc)})
    matches=[x['path'] for x in report['manifest_candidates'] if x.get('matches_expected')]
    report['matching_manifest_paths']=matches
    report['status']='MATCH_FOUND_REVIEW_CONTENT' if matches else 'STOP_REVIEW_ORIGINAL_NOT_FOUND'
    report['limitations']=['Hash equality alone does not establish original split provenance or data leakage absence.','No checkpoint weights or manifest are modified.','No training, validation or official test is performed.','Only runs/ and existing backup/, backups/, archive/, archives/ directories are searched.']
    path=out/'h34_result.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('H34 RESULT:',json.dumps(report,ensure_ascii=False))
    print('H34 COMPLETE:',out)
    print('NEXT: share H34 RESULT; do not replace or edit manifest to force a hash match.')
if __name__=='__main__':main()
