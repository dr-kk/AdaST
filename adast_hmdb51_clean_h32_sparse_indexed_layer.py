#!/usr/bin/env python3
"""H32 synthetic sparse indexed TimeSformer layer probe; no data/checkpoints/training.
Uses genuine HF layer modules with original frame/spatial indices, retains CLS.
Compares full occupancy to native HF and sparse indexed output to independent
per-token grouped-attention reference. No accuracy/speedup claims.
"""
import argparse, inspect, json
from pathlib import Path
import torch
import transformers
from transformers import TimesformerConfig
from transformers.models.timesformer.modeling_timesformer import TimesformerLayer


def indexed(layer, h, ids, frames):
    assert h.shape[0] == 1 and h.shape[1] == len(ids)+1 and ids.ndim == 2 and ids.shape[1] == 2
    assert len(ids) and len({tuple(x) for x in ids.tolist()}) == len(ids)
    assert (ids[:,0] >= 0).all() and (ids[:,0] < frames).all() and (ids[:,1] >= 0).all()
    assert layer.attention_type == 'divided_space_time' and not layer.training
    patches = h[:,1:,:]
    temporal = torch.empty_like(patches)
    # Temporal attention: only tokens sharing an ORIGINAL spatial coordinate.
    for s in torch.unique(ids[:,1], sorted=True).tolist():
        ix = torch.nonzero(ids[:,1] == s, as_tuple=True)[0]
        ix = ix[torch.argsort(ids[ix,0])]
        temporal[:,ix,:] = layer.temporal_attention(layer.temporal_layernorm(patches[:,ix,:]))[0]
    temporal_embedding = patches + layer.temporal_dense(layer.drop_path(temporal))
    cls = h[:,:1,:]
    residual = torch.zeros_like(temporal_embedding)
    cls_parts = []
    # Spatial attention: one CLS per original frame, including empty frames.
    for t in range(frames):
        ix = torch.nonzero(ids[:,0] == t, as_tuple=True)[0]
        ix = ix[torch.argsort(ids[ix,1])]
        z = torch.cat([cls, temporal_embedding[:,ix,:]], dim=1)
        att = layer.attention(layer.layernorm_before(z))[0]
        cls_parts.append(att[:,:1,:])
        if ix.numel(): residual[:,ix,:] = att[:,1:,:]
    cls_residual = layer.drop_path(torch.cat(cls_parts, dim=0)).mean(dim=0, keepdim=True)
    spatial_residual = layer.drop_path(residual)
    combined = torch.cat([cls,temporal_embedding],dim=1) + torch.cat([cls_residual,spatial_residual],dim=1)
    mlp = layer.output(layer.intermediate(layer.layernorm_after(combined)))
    return combined + layer.drop_path(mlp)


def independent_reference(layer, h, ids, frames):
    """Per-token reference: recompute each group and extract only queried token.
    Deliberately slower and separate assembly from indexed().
    """
    patches = h[:,1:,:]
    tmp = []
    for j in range(len(ids)):
        s = int(ids[j,1]); members = [k for k in range(len(ids)) if int(ids[k,1]) == s]
        members.sort(key=lambda k:int(ids[k,0]))
        group = patches[:,members,:]
        attended = layer.temporal_attention(layer.temporal_layernorm(group))[0]
        tmp.append(attended[:,members.index(j):members.index(j)+1,:])
    temporal = patches + layer.temporal_dense(layer.drop_path(torch.cat(tmp,dim=1)))
    cls = h[:,:1,:]
    spatial_patches = []
    cls_parts = []
    for t in range(frames):
        members = [k for k in range(len(ids)) if int(ids[k,0]) == t]
        members.sort(key=lambda k:int(ids[k,1]))
        z = torch.cat([cls,temporal[:,members,:]],dim=1)
        att = layer.attention(layer.layernorm_before(z))[0]
        cls_parts.append(att[:,:1,:])
    for j in range(len(ids)):
        t = int(ids[j,0]); members = [k for k in range(len(ids)) if int(ids[k,0]) == t]
        members.sort(key=lambda k:int(ids[k,1]))
        z = torch.cat([cls,temporal[:,members,:]],dim=1)
        att = layer.attention(layer.layernorm_before(z))[0]
        spatial_patches.append(att[:,members.index(j)+1:members.index(j)+2,:])
    cls_delta = layer.drop_path(torch.cat(cls_parts,dim=0)).mean(dim=0,keepdim=True)
    patch_delta = layer.drop_path(torch.cat(spatial_patches,dim=1))
    merged = torch.cat([cls,temporal],dim=1)+torch.cat([cls_delta,patch_delta],dim=1)
    return merged + layer.drop_path(layer.output(layer.intermediate(layer.layernorm_after(merged))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default=r'runs\hmdb51_clean_h32_sparse_indexed_s1')
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args();torch.manual_seed(a.seed)
    config=TimesformerConfig(image_size=32,patch_size=16,num_frames=8,hidden_size=48,
        num_attention_heads=4,intermediate_size=96,num_hidden_layers=1,
        attention_type='divided_space_time',hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,drop_path_rate=0)
    layer=TimesformerLayer(config,0).eval();frames=8;spatial=4
    full_ids=torch.tensor([(t,s) for s in range(spatial) for t in range(frames)],dtype=torch.long)
    full_h=torch.randn(1,1+len(full_ids),48)
    # Nonuniform per-frame spatial occupancy; one frame is empty, others differ.
    keep=[(t,s) for s in range(spatial) for t in range(frames)
          if (t!=5 and ((s+t)%3!=0))]
    ids=torch.tensor(keep,dtype=torch.long)
    sparse_h=torch.cat([full_h[:,:1,:],full_h[:,1:,:][:,torch.tensor([s*frames+t for t,s in keep]),:]],dim=1)
    with torch.inference_mode():
        native=layer(full_h)[0]
        full=indexed(layer,full_h,full_ids,frames)
        sparse=indexed(layer,sparse_h,ids,frames)
        ref=independent_reference(layer,sparse_h,ids,frames)
    full_err=float((native-full).abs().max())
    sparse_err=float((sparse-ref).abs().max())
    full_ok=bool(torch.allclose(native,full,atol=1e-5,rtol=1e-5))
    sparse_ok=bool(torch.allclose(sparse,ref,atol=1e-5,rtol=1e-5))
    result=dict(experiment='H32 synthetic sparse indexed genuine TimeSformer layer',
        status='PASS' if full_ok and sparse_ok else 'FAIL',torch_version=torch.__version__,
        transformers_version=transformers.__version__,layer_forward_signature=str(inspect.signature(TimesformerLayer.forward)),
        full_token_equivalence=dict(pass_check=full_ok,max_abs_error=full_err),
        sparse_indexed_reference=dict(pass_check=sparse_ok,max_abs_error=sparse_err),
        original_patch_tokens=len(full_ids),retained_patch_tokens=len(ids),
        retained_per_frame=[int((ids[:,0]==t).sum()) for t in range(frames)],
        original_sequence_length=full_h.shape[1],compact_sequence_length=sparse_h.shape[1],
        output_shape=list(sparse.shape),original_indices_preserved=True,
        pretrained_checkpoint_loaded=False,dataset_used=False,training_performed=False,
        validation_evaluated=False,official_test_evaluated=False,
        limitations=['Synthetic randomly initialized single HF layer, not an AdaST checkpoint.',
        'Sparse reference uses the same HF attention modules, not an independent attention-kernel implementation.',
        'Pruned sparse outputs need not equal full-model outputs; attention context changes.',
        'No complete backbone, token controller/consolidator, position-embedding adaptation, accuracy or latency benchmark.',
        'Python grouping and repeated reference forwards are not optimized for speed.'])
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise SystemExit(f'Output folder not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('H32 RESULT:',json.dumps(result),flush=True)
    print('H32 COMPLETE:',out.resolve(),flush=True)
    if result['status']!='PASS':raise SystemExit('H32 FAILED; stop before checkpoint integration')

if __name__=='__main__':main()
