#!/usr/bin/env python3
"""H30: synthetic, position-indexed physical token compaction feasibility probe.
No dataset, checkpoints, network, training or official-test access. CPU by default.
This is a new attention prototype, NOT a drop-in TimeSformer/AdaST replacement.
"""
import argparse, json, math, time
from pathlib import Path
import torch
from torch import nn


def compact(x, keep):
    # x [T,N,D], keep [T,N]; preserve original (frame, spatial-position) metadata.
    ids = keep.nonzero(as_tuple=False)
    if not len(ids):
        raise ValueError('No retained tokens')
    return x[ids[:, 0], ids[:, 1]], ids


def grouped_attention(qkv, group_ids, scale):
    """Attention only within matching group IDs, with explicit original indices."""
    out = torch.empty_like(qkv)
    for group in torch.unique(group_ids, sorted=True):
        ix = (group_ids == group).nonzero(as_tuple=True)[0]
        z = qkv.index_select(0, ix)
        scores = (z @ z.T) * scale
        out.index_copy_(0, ix, scores.softmax(dim=-1) @ z)
    return out


class IndexedDividedAttention(nn.Module):
    """Tiny illustrative divided attention, not HF's trained attention implementation."""
    def __init__(self, dim):
        super().__init__()
        self.temporal = nn.Linear(dim, dim, bias=False)
        self.spatial = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, compact_tokens, original_indices):
        if compact_tokens.ndim != 2 or original_indices.shape != (len(compact_tokens), 2):
            raise ValueError('Invalid token/index shapes')
        # Same original spatial position across frames; no accidental positional mixing.
        temporal_out = grouped_attention(self.temporal(compact_tokens), original_indices[:, 1], self.scale)
        # Same original frame across spatial positions.
        spatial_out = grouped_attention(self.spatial(temporal_out), original_indices[:, 0], self.scale)
        return spatial_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=r'runs\hmdb51_clean_h30_indexed_pruning_s1')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    T, N, D = 8, 16, 48
    x = torch.randn(T, N, D)
    # Different spatial positions retained per frame; frame count stays 8.
    keep = torch.zeros(T, N, dtype=torch.bool)
    for t in range(T):
        keep[t, [(t + j * 3) % N for j in range(6)]] = True
    compact_x, idx = compact(x, keep)
    layer = IndexedDividedAttention(D).eval()
    with torch.no_grad():
        y = layer(compact_x, idx)
        # Independent reference: explicit selection of groups and matrix products.
        z = layer.temporal(compact_x)
        ref_t = torch.zeros_like(z)
        for i in range(len(idx)):
            members = torch.where(idx[:, 1] == idx[i, 1])[0]
            vals = z[members]
            weights = (z[i] @ vals.T * layer.scale).softmax(dim=0)
            ref_t[i] = weights @ vals
        z2 = layer.spatial(ref_t)
        ref_y = torch.zeros_like(z2)
        for i in range(len(idx)):
            members = torch.where(idx[:, 0] == idx[i, 0])[0]
            vals = z2[members]
            weights = (z2[i] @ vals.T * layer.scale).softmax(dim=0)
            ref_y[i] = weights @ vals
    err = (y - ref_y).abs().max().item()
    assert y.shape == (int(keep.sum()), D)
    assert torch.equal(compact_x, x[idx[:, 0], idx[:, 1]])
    assert err < 1e-5, f'Indexed grouping mismatch: {err}'
    # Demonstrate why naive per-frame reindexing is semantically wrong:
    # local slot zero refers to different original positions across frames.
    first_positions = idx[idx[:, 1] == 0, 1]  # not used for the assertion below
    local_slot_zero = [int(torch.where(keep[t])[0][0]) for t in range(T)]
    assert len(set(local_slot_zero)) > 1
    summary = {
        'experiment': 'H30 synthetic indexed physical token compaction and grouped attention',
        'status': 'PASS', 'torch_version': torch.__version__,
        'original_patch_tokens': T*N, 'retained_patch_tokens': len(idx),
        'physical_sequence_reduction_pct': 100*(1-len(idx)/(T*N)),
        'frames': T, 'original_patches_per_frame': N,
        'retained_per_frame': [int(keep[t].sum()) for t in range(T)],
        'max_abs_reference_error': err,
        'local_slot_zero_original_positions': local_slot_zero,
        'dataset_used': False, 'checkpoint_loaded': False, 'training_performed': False,
        'validation_evaluated': False, 'official_test_evaluated': False,
        'limitations': [
            'Prototype attention is not pretrained TimeSformer attention and does not load AdaST weights.',
            'This is not a validated accuracy-preserving architecture or a speed benchmark.',
            'Python group loops and controller/consolidator overhead may outweigh reduced attention work.',
            'Temporal attention groups only surviving tokens at identical original spatial positions.',
            'CLS, positional embeddings, residual/MLP, original attention weights, and consolidation are not integrated.'
        ]
    }
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('H30 RESULT:', json.dumps(summary))
    print('H30 COMPLETE:', out.resolve())

if __name__ == '__main__':
    main()
