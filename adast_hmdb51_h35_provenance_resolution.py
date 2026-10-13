#!/usr/bin/env python3
"""H35: read-only checkpoint/manifest provenance inspection. Does not certify a split."""
import argparse, hashlib, json
from pathlib import Path
import torch

def sha(b): return hashlib.sha256(b).hexdigest()
def load_checkpoint(path):
    x=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(x,dict): raise ValueError('Checkpoint is not a dictionary')
    return x

def summarize(x,depth=0):
    if depth>2:return type(x).__name__
    if isinstance(x,dict):
        return {str(k):summarize(v,depth+1) for k,v in x.items() if str(k) not in ('model_state_dict','state_dict','model','optimizer','optimizer_state_dict','scheduler','scaler','rng_state')}
    if isinstance(x,(list,tuple)):return {'type':type(x).__name__,'length':len(x),'first_items':[summarize(v,depth+1) for v in x[:2]]}
    if isinstance(x,(str,int,float,bool)) or x is None:return x if not isinstance(x,str) or len(x)<240 else x[:240]+'...'
    if isinstance(x,torch.Tensor):return {'tensor_shape':list(x.shape),'dtype':str(x.dtype)}
    return type(x).__name__

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense-checkpoint',required=True)
    p.add_argument('--h13-checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():p.error('Output already exists; use a new directory')
    paths={k:Path(v) for k,v in [('dense',a.dense_checkpoint),('h13',a.h13_checkpoint),('manifest',a.manifest)]}
    for name,path in paths.items():
        if not path.is_file():p.error(f'{name} not found: {path}')
    raw=paths['manifest'].read_bytes(); manifest=json.loads(raw.decode('utf-8-sig'))
    ck={name:load_checkpoint(paths[name]) for name in ('dense','h13')}
    hashes={name:obj.get('manifest_sha256') for name,obj in ck.items()}
    canonical_variants={}
    for indent in (None,2,4):
        for sort in (False,True):
            for ascii_flag in (False,True):
                for newline in ('','\n'):
                    label=f'indent={indent};sort={sort};ascii={ascii_flag};newline={bool(newline)}'
                    b=(json.dumps(manifest,indent=indent,sort_keys=sort,ensure_ascii=ascii_flag)+newline).encode('utf-8')
                    if sha(b) in hashes.values():canonical_variants[label]=sha(b)
    split_info={k:{'type':type(manifest.get(k)).__name__,'count':len(manifest.get(k,[])) if isinstance(manifest.get(k),list) else None} for k in ('train','val')}
    overlap=None
    try:
        def identifier(x):return json.dumps(x,sort_keys=True,ensure_ascii=False)
        overlap=len(set(map(identifier,manifest['train'])) & set(map(identifier,manifest['val'])))
    except (KeyError,TypeError,ValueError):pass
    result={'experiment':'H35 read-only split provenance resolution','status':'ORIGINAL_BYTES_VERIFIED' if sha(raw) in hashes.values() else 'UNRESOLVED',
      'expected_checkpoint_manifest_hashes':hashes,'current_manifest_sha256':sha(raw),'canonical_serialization_matches':canonical_variants,
      'manifest_fields':list(manifest),'split_info':split_info,'train_val_exact_record_overlap':overlap,
      'checkpoint_metadata':{name:summarize(obj) for name,obj in ck.items()},
      'notes':['Canonical serialization matching a hash is a clue, not proof that records were unchanged.',
      'Checkpoint metadata may contain paths or user-supplied arguments; inspect locally before sharing.',
      'No dataset, training, validation or official test used; no files modified except new output JSON.']}
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    safe={k:v for k,v in result.items() if k!='checkpoint_metadata'}
    safe['checkpoint_metadata_keys']={name:list(obj) for name,obj in ck.items()}
    print('H35 RESULT:',json.dumps(safe,ensure_ascii=False),flush=True)
    print('H35 COMPLETE:',out.resolve(),flush=True)
    print('H35 NOTE: Inspect summary.json locally for checkpoint metadata; share only non-sensitive provenance fields.',flush=True)
if __name__=='__main__':main()
