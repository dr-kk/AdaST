#!/usr/bin/env python3
"""H33-A: read-only checkpoint compatibility audit; no dataset, model forward or test."""
import argparse, hashlib, json
from pathlib import Path
import torch

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def unpack(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(obj,dict):raise ValueError(f'Checkpoint is not a dict: {path}')
    for key in ('model_state_dict','state_dict','model'):
        if isinstance(obj.get(key),dict):return obj,obj[key]
    raise ValueError(f'No state dict in {path}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense-checkpoint',required=True)
    p.add_argument('--h13-checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',default=r'runs\hmdb51_clean_h33_checkpoint_audit_s1')
    a=p.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):p.error('Output directory is not empty; choose a new output')
    dense_path=Path(a.dense_checkpoint);h13_path=Path(a.h13_checkpoint);manifest_path=Path(a.manifest)
    for path in (dense_path,h13_path,manifest_path):
        if not path.is_file():p.error(f'Missing file: {path}')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    dense,dense_sd=unpack(dense_path);h13,h13_sd=unpack(h13_path)
    dense_keys={k[7:] if k.startswith('module.') else k:v for k,v in dense_sd.items()}
    h13_keys={k[7:] if k.startswith('module.') else k:v for k,v in h13_sd.items()}
    backbone={k[len('backbone.'):]:v for k,v in h13_keys.items() if k.startswith('backbone.')}
    missing=sorted(set(dense_keys)-set(backbone));extra=sorted(set(backbone)-set(dense_keys))
    shape_mismatch=sorted(k for k in set(dense_keys)&set(backbone) if dense_keys[k].shape!=backbone[k].shape)
    differing=sorted(k for k in set(dense_keys)&set(backbone) if dense_keys[k].shape==backbone[k].shape and not torch.equal(dense_keys[k],backbone[k]))
    digest=sha(manifest_path)
    manifest_hashes={name:obj.get('manifest_sha256') for name,obj in [('dense',dense),('h13',h13)]}
    policy={prefix:sum(k.startswith(prefix) for k in h13_keys) for prefix in ('scorer.','budget.','temporal_budget.','consolidator.')}
    statuses={
      'manifest_matches_dense':manifest_hashes['dense']==digest,
      'manifest_matches_h13':manifest_hashes['h13']==digest,
      'backbone_keys_match':not missing and not extra and not shape_mismatch,
      'backbone_weights_identical':not differing and not missing and not extra and not shape_mismatch,
      'policy_modules_present':all(policy.values()),
      'clean_split_counts':len(manifest.get('train',[]))==3213 and len(manifest.get('val',[]))==357,
    }
    result={'experiment':'H33-A read-only trained checkpoint compatibility audit',
      'status':'PASS' if all(statuses.values()) else 'STOP_REVIEW',
      'checks':statuses,'manifest_sha256':digest,'checkpoint_sha256':{'dense':sha(dense_path),'h13':sha(h13_path)},
      'checkpoint_manifest_hashes':manifest_hashes,'dense_keys':len(dense_keys),'h13_backbone_keys':len(backbone),
      'h13_policy_key_counts':policy,'missing_backbone_keys':missing[:15],'extra_backbone_keys':extra[:15],
      'shape_mismatches':shape_mismatch[:15],'different_backbone_weights_count':len(differing),
      'different_backbone_weights_examples':differing[:15],
      'training_performed':False,'validation_evaluated':False,'official_test_evaluated':False,
      'limitations':['No model instantiated or forward pass performed.','Does not prove sparse attention integration, accuracy or speedup.','A modified sparse forward changes attention context and requires separate validation.']}
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('H33-A RESULT:',json.dumps(result),flush=True)
    print('H33-A COMPLETE:',out.resolve(),flush=True)
    if result['status']!='PASS':raise SystemExit('STOP: resolve checkpoint/manifest mismatch before integration')
if __name__=='__main__':main()
