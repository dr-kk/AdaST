#!/usr/bin/env python
"""H36-B: read-only trained H13 sparse downstream integration probe.
Synthetic input only. Indexed HF downstream is a correctness prototype, NOT a speed benchmark.
"""
import argparse, hashlib, json, sys, types
from pathlib import Path
import torch
from transformers import TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h32_sparse_indexed_layer import indexed

EXPECTED='8b45fc6c63ea8602becd674f59f1f8892174ce352c46fe45a7745423f8405bec'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--h13-checkpoint',default=r'runs\hmdb51_clean_h13_balanced_sampling_s1\best_val.pt')
    ap.add_argument('--manifest',default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    ap.add_argument('--output',default=r'runs\hmdb51_clean_h36b_trained_sparse_s1')
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); out=Path(a.output)
    if out.exists() and any(out.iterdir()):ap.error('Output directory must be absent or empty')
    m=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    digest=hashlib.sha256(json.dumps(m,indent=2,ensure_ascii=False).encode('utf-8')).hexdigest()
    if digest!=EXPECTED or len(m['train'])!=3213 or len(m['val'])!=357:raise RuntimeError('Manifest verification failed')
    ck=torch.load(a.h13_checkpoint,map_location='cpu',weights_only=False)
    if ck.get('manifest_sha256')!=EXPECTED or ck.get('split')!=1 or ck.get('num_frames')!=8 or ck.get('protocol')!='Clean H13 balanced sampling; official test not evaluated':raise RuntimeError('Checkpoint protocol mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(a.seed)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.to(device).eval()
    layers=model.layers(); original_forward={i:layers[i].forward for i in range(6,len(layers))}
    capture={}
    # Registered AFTER AdaSTE5's controller hook, so this receives its actual consolidated output.
    def inspect_hook(module,args,kwargs):
        h=args[0] if args else kwargs['hidden_states']
        if h.shape[0]!=1 or (h.shape[1]-1)%8:raise RuntimeError('Unsupported sequence geometry')
        capture['full']=h.detach().clone()
        return None
    handle=layers[6].register_forward_pre_hook(inspect_hook,with_kwargs=True)
    x=torch.randn(1,8,3,224,224,device=device)
    try:
        with torch.inference_mode():
            baseline=model(x).logits.detach().float().cpu()
            full=capture['full']; n=(full.shape[1]-1)//8
            # Genuine trained layer-6: full-token indexed forward vs original HF forward.
            full_ids=torch.tensor([(j%8,j//8) for j in range(full.shape[1]-1)],dtype=torch.long)
            native=original_forward[6](full)[0]
            indexed_full=indexed(layers[6],full,full_ids,8)
            full_err=float((native-indexed_full).abs().max().item())
            full_ok=bool(torch.allclose(native,indexed_full,atol=2e-4,rtol=2e-4))
            # H13 consolidator leaves dropped slots zero; verify exact count against K*S.
            mask=full[:,1:,:].abs().amax(-1)[0]>0
            positions=torch.nonzero(mask,as_tuple=True)[0]
            k=int(model.last_k[0]); s=int(model.last_s[0]); expected=k*s
            if len(positions)!=expected:raise RuntimeError(f'Nonzero retained slots {len(positions)} != controller K*S {expected}; cannot infer selection safely')
            ids=torch.stack([positions%8,positions//8],dim=1).cpu()
            if len(positions)==0 or len(positions)>=8*n:raise RuntimeError('Synthetic input did not select a proper sparse subset; change seed')
            compact=torch.cat([full[:,:1,:],full[:,1:,:][:,positions,:]],dim=1)
            capture['compact']=compact; capture['ids']=ids
            def compact_hook(module,args,kwargs):
                h=args[0] if args else kwargs['hidden_states']
                if h.shape!=capture['full'].shape:raise RuntimeError('Unexpected block-6 input')
                if args:return (capture['compact'],)+args[1:],kwargs
                z=dict(kwargs);z['hidden_states']=capture['compact'];return args,z
            handle.remove();handle=None
            compact_handle=layers[6].register_forward_pre_hook(compact_hook,with_kwargs=True)
            try:
                for i in range(6,len(layers)):
                    def make_forward(layer):
                        def sparse_forward(self,hidden_states,output_attentions=False):
                            if output_attentions:raise RuntimeError('Attention maps unsupported in sparse probe')
                            if hidden_states.shape[1]!=compact.shape[1]:raise RuntimeError('Compact token count changed')
                            return (indexed(self,hidden_states,ids,8),)
                        return types.MethodType(sparse_forward,layer)
                    layers[i].forward=make_forward(layers[i])
                sparse=model(x).logits.detach().float().cpu()
            finally:
                compact_handle.remove()
                for i,fn in original_forward.items():layers[i].forward=fn
            # Re-evaluate original model after restoration: detects unintentional mutation.
            restored=model(x).logits.detach().float().cpu()
            restored_err=float((baseline-restored).abs().max().item())
    finally:
        if handle is not None:handle.remove()
    result=dict(experiment='H36-B trained H13 indexed sparse downstream integration',
        status='PASS' if full_ok and restored_err<=1e-5 and bool(torch.isfinite(sparse).all()) and sparse.shape==baseline.shape else 'FAIL',
        device=str(device),manifest_canonical_sha256=digest,checkpoint_loaded_strictly=True,
        trained_layer6_full_token_equivalence=dict(pass_check=full_ok,max_abs_error=full_err),
        original_model_restored_logit_max_abs_error=restored_err,
        original_patch_tokens=8*n,retained_patch_tokens=len(positions),k_per_frame=k,selected_frames=s,
        original_downstream_sequence_length=full.shape[1],actual_compact_sequence_length=compact.shape[1],
        sparse_forward_logits_finite=bool(torch.isfinite(sparse).all()),
        sparse_vs_original_logit_max_abs_difference=float((baseline-sparse).abs().max().item()),
        sparse_vs_original_prediction_equal=bool(baseline.argmax(-1).item()==sparse.argmax(-1).item()),
        synthetic_input_only=True,training_performed=False,validation_evaluated=False,official_test_evaluated=False,
        limitations=['Indexed downstream is a Python-grouped prototype, not optimized for speed.',
          'Sparse and dense predictions are not expected to match: attention context changes.',
          'Full-token equivalence checked on trained layer 6, not every downstream layer or the modified full model.',
          'Retention positions inferred from zero slots; checked against controller K*S for this input only.',
          'No dataset accuracy, measured latency, GPU-memory benefit or FLOP reduction established.'])
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('H36-B RESULT:',json.dumps(result),flush=True)
    print('H36-B COMPLETE:',out.resolve(),flush=True)
    if result['status']!='PASS':sys.exit(2)
if __name__=='__main__':main()
