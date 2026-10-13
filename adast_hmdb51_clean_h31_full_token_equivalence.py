#!/usr/bin/env python3
"""H31: exact-weight, full-token equivalence of an indexed TimeSformer layer.
Synthetic CPU only. No datasets, checkpoints, downloads, or training.
Tests the installed HF divided-space-time layer against a position-indexed
implementation using the SAME layer's modules and weights. Does not establish
pruned-mode accuracy, latency, or pretrained-checkpoint equivalence.
"""
import argparse, json, inspect
from pathlib import Path
import torch
from transformers import TimesformerConfig
from transformers.models.timesformer.modeling_timesformer import TimesformerLayer


def indexed_forward(layer, h, original_ids, frames, spatial_positions):
    """Full-token implementation: original_ids [T*N,2] in HF's native order.
    Temporal grouping uses original spatial index, spatial grouping original frame.
    This function deliberately requires full, rectangular occupancy; sparse mode
    needs a separate semantics/CLS integration design and is NOT implemented here.
    """
    b, length, d = h.shape
    assert b == 1 and length == 1 + frames * spatial_positions
    assert original_ids.shape == (frames * spatial_positions, 2)
    assert torch.equal(original_ids, torch.tensor([(t, s) for s in range(spatial_positions) for t in range(frames)], device=h.device))
    assert layer.attention_type == 'divided_space_time'
    patches = h[:, 1:, :]
    temporal_groups = []
    for s in range(spatial_positions):
        ix = torch.nonzero(original_ids[:, 1] == s, as_tuple=True)[0]
        temporal_groups.append(layer.temporal_attention(layer.temporal_layernorm(patches[:, ix, :]))[0])
    temporal_residual = torch.cat(temporal_groups, dim=1)
    temporal_residual = layer.temporal_dense(layer.drop_path(temporal_residual))
    temporal_embedding = patches + temporal_residual
    init_cls = h[:, :1, :]
    cls_by_frame = init_cls.repeat(1, frames, 1).reshape(frames, 1, d)
    spatial_groups = []
    for t in range(frames):
        ix = torch.nonzero(original_ids[:, 0] == t, as_tuple=True)[0]
        z = torch.cat([cls_by_frame[t:t+1], temporal_embedding[:, ix, :]], dim=1)
        spatial_groups.append(layer.attention(layer.layernorm_before(z))[0])
    residual_spatial = layer.drop_path(torch.cat([z[:, 1:, :] for z in spatial_groups], dim=0))
    cls_residual = layer.drop_path(torch.cat([z[:, :1, :] for z in spatial_groups], dim=0)).mean(0, keepdim=True)
    # Spatial group outputs are frame-major. Restore native HF order: spatial-major, frame-minor.
    spatial_patches = residual_spatial.reshape(frames, spatial_positions, d).permute(1, 0, 2).reshape(1, frames * spatial_positions, d)
    combined = torch.cat([init_cls, temporal_embedding], dim=1) + torch.cat([cls_residual, spatial_patches], dim=1)
    z = layer.layernorm_after(combined)
    z = layer.intermediate(z)
    z = layer.output(z)
    return combined + layer.drop_path(z)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=r'runs\hmdb51_clean_h31_full_token_equivalence_s1')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    config = TimesformerConfig(image_size=32, patch_size=16, num_frames=8, hidden_size=48,
        num_attention_heads=4, intermediate_size=96, num_hidden_layers=1,
        attention_type='divided_space_time', hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, drop_path_rate=0.0)
    layer = TimesformerLayer(config, 0).eval()
    frames, spatial = 8, 4
    ids = torch.tensor([(t, s) for s in range(spatial) for t in range(frames)], dtype=torch.long)
    h = torch.randn(1, 1 + frames * spatial, config.hidden_size)
    with torch.inference_mode():
        reference = layer(h)[0]
        indexed = indexed_forward(layer, h, ids, frames, spatial)
    err = float((reference-indexed).abs().max())
    status = 'PASS' if torch.allclose(reference, indexed, atol=1e-5, rtol=1e-5) else 'FAIL'
    result = dict(experiment='H31 exact-weight full-token indexed-layer equivalence', status=status,
        torch_version=torch.__version__, transformers_version=__import__('transformers').__version__,
        layer_signature=str(inspect.signature(TimesformerLayer.forward)),
        input_shape=list(h.shape), output_shape=list(indexed.shape), max_abs_error=err,
        full_token_reference_equivalent=status=='PASS', pretrained_checkpoint_loaded=False,
        physical_pruning_implemented=False, dataset_used=False, validation_evaluated=False,
        official_test_evaluated=False, training_performed=False,
        limitations=['Randomly initialized genuine HF TimeSformerLayer, not pretrained AdaST weights.',
        'Full-token occupancy only; sparse indexed attention and CLS handling not yet implemented.',
        'No model-level checkpoint integration, accuracy, memory or speedup established.'])
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Output folder is not empty: {out}; choose a new output folder')
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('H31 RESULT:', json.dumps(result), flush=True)
    print('H31 COMPLETE:', out.resolve(), flush=True)
    if status != 'PASS': raise SystemExit('H31 equivalence FAILED; do not proceed to pruning.')

if __name__ == '__main__': main()
