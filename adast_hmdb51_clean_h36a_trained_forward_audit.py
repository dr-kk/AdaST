#!/usr/bin/env python
"""H36-A: trained H13 forward/selection audit; no dataset or official test.
Checks a separately instantiated model with a read-only block-6 probe against the
unmodified H13 model. Does NOT implement physical sparse downstream inference.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from transformers import TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME

EXPECTED = '8b45fc6c63ea8602becd674f59f1f8892174ce352c46fe45a7745423f8405bec'

def sha_bytes(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def load_model(path, device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('manifest_sha256')!=EXPECTED:raise RuntimeError('H13 checkpoint manifest hash mismatch')
    if ck.get('split')!=1 or ck.get('num_frames')!=8:raise RuntimeError('H13 split/frame metadata mismatch')
    if ck.get('protocol')!='Clean H13 balanced sampling; official test not evaluated':raise RuntimeError('Unexpected H13 protocol')
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    return model.to(device).eval()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h13-checkpoint',default=r'runs\hmdb51_clean_h13_balanced_sampling_s1\best_val.pt')
    p.add_argument('--manifest',default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    p.add_argument('--output',default=r'runs\hmdb51_clean_h36a_trained_forward_s1')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args(); out=Path(a.output)
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty or absent')
    manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    canonical=json.dumps(manifest,indent=2,ensure_ascii=False).encode('utf-8')
    if hashlib.sha256(canonical).hexdigest()!=EXPECTED:raise RuntimeError('Canonical manifest hash mismatch; stop')
    if len(manifest['train'])!=3213 or len(manifest['val'])!=357:raise RuntimeError('Unexpected split counts')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.manual_seed(a.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(a.seed)
    model=load_model(a.h13_checkpoint,device)
    # Hook registration order: AdaSTE5's controller hook runs first, then this read-only probe.
    observed={}
    def probe(module,args,kwargs):
        h=args[0] if args else kwargs['hidden_states']
        b,seq,d=h.shape; t=model.T; n=(seq-1)//t
        if b!=1 or (seq-1)%t:raise RuntimeError('Unexpected block-6 sequence shape')
        # Shape-preserving consolidation zeros dropped slots; count is NOT a
        # reliable physical-retention measure if a retained feature is zero.
        observed['block6_input_shape']=list(h.shape)
        observed['nonzero_patch_slots_after_consolidation']=int((h[:,1:].abs().amax(-1)>0).sum().item())
        observed['spatial_patch_slots_per_frame']=n
        return None
    handle=model.layers()[6].register_forward_pre_hook(probe,with_kwargs=True)
    # One deterministic synthetic video; not a validation/test sample.
    x=torch.randn(1,8,3,224,224,device=device)
    try:
        with torch.inference_mode():
            original=model(x).logits.float().cpu()
            first={k:v.detach().cpu().clone() for k,v in (
                ('k',model.last_k),('s',model.last_s),('r',model.last_r),
                ('actual_r',model.last_actual_r))}
            # Probe is read-only. Repeated forward must reproduce logits and policy.
            repeated=model(x).logits.float().cpu()
            second={k:v.detach().cpu().clone() for k,v in (
                ('k',model.last_k),('s',model.last_s),('r',model.last_r),
                ('actual_r',model.last_actual_r))}
    finally:handle.remove()
    err=float((original-repeated).abs().max().item())
    policies_equal=all(torch.equal(first[k],second[k]) for k in first)
    k=int(first['k'][0]);s=int(first['s'][0]);n=observed['spatial_patch_slots_per_frame']
    expected_kept=k*s
    # Exact joint keep count follows the controller's K-per-frame and S-frame choices.
    # Original H13 still feeds all patch slots into the HF downstream backbone.
    result=dict(experiment='H36-A trained H13 deterministic forward and sparse-budget audit',
        status='PASS' if err<=1e-5 and policies_equal and 1<=k<=n and 6<=s<=8 else 'FAIL',
        device=str(device),checkpoint_sha256=sha_bytes(a.h13_checkpoint),
        canonical_manifest_sha256=hashlib.sha256(canonical).hexdigest(),
        synthetic_input_only=True,checkpoint_loaded_strictly=True,
        repeated_forward_max_abs_logit_error=err,policy_decisions_equal=policies_equal,
        block6_input_shape=observed['block6_input_shape'],
        original_patch_tokens=n*8,controller_selected_patch_tokens=expected_kept,
        controller_selected_fraction=expected_kept/(n*8),
        k_per_frame=k,selected_frames=s,
        original_downstream_sequence_length=n*8+1,
        hypothetical_compact_sequence_length=expected_kept+1,
        physical_sparse_downstream_implemented=False,
        validation_evaluated=False,official_test_evaluated=False,
        limitations=['Read-only probe and repeated-forward check, NOT an independent modified-backbone equivalence test.',
                     'Selected token count is a controller budget; downstream sequence is NOT compacted.',
                     'No accuracy, latency or memory improvement measured.'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('H36-A RESULT:',json.dumps(result),flush=True)
    print('H36-A COMPLETE:',out.resolve(),flush=True)
    if result['status']!='PASS':sys.exit(2)
if __name__=='__main__':main()
