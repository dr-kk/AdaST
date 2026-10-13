#!/usr/bin/env python
"""H29-B: CPU-only synthetic TimeSformer block shape probe. No data/checkpoints/test."""
import argparse
import inspect
import json
from pathlib import Path
import torch
import transformers
from transformers import TimesformerConfig, TimesformerForVideoClassification


def attempt(layer, tokens, label, width, frames, dim):
    # CLS + synthetic patch tokens. No actual video or checkpoint is accessed.
    x = torch.randn(1, 1 + tokens, dim)
    result = dict(case=label, patch_tokens=tokens, input_shape=list(x.shape),
                  inferred_spatial_tokens=tokens // frames,
                  inferred_height=(tokens // frames) // width,
                  expected_reshape_patch_tokens=((tokens // frames) // width) * width * frames)
    try:
        with torch.no_grad():
            y = layer(x)[0]
        result.update(status='PASS', output_shape=list(y.shape),
                      sequence_length_preserved=(y.shape[1] == x.shape[1]))
    except Exception as exc:
        result.update(status='FAIL', error_type=type(exc).__name__, error=str(exc)[:1000])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=r'runs\hmdb51_clean_h29b_synthetic_shape_s1')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        p.error(f'Output folder is not empty: {out}')
    # Tiny randomly initialized model: structure matches divided-space-time attention,
    # but this is NOT the user's trained model and cannot establish accuracy or speedup.
    frames, image, patch, dim, width = 8, 32, 16, 48, 2
    config = TimesformerConfig(image_size=image, patch_size=patch, num_frames=frames,
                               hidden_size=dim, num_hidden_layers=1, num_attention_heads=4,
                               intermediate_size=96, attention_type='divided_space_time',
                               num_labels=2)
    model = TimesformerForVideoClassification(config).eval().cpu()
    layer = model.timesformer.encoder.layer[0]
    results = [
        attempt(layer, 8 * 4, 'A_full_4_patches_per_frame', width, frames, dim),
        attempt(layer, 8 * 3, 'B_compact_3_patches_per_frame', width, frames, dim),
        attempt(layer, 8 * 2, 'C_structured_2_patches_per_frame', width, frames, dim),
        attempt(layer, 6 * 4, 'D_drop_2_frames_original_4_patches', width, frames, dim),
    ]
    summary = dict(experiment='Clean H29-B synthetic TimeSformer shape compatibility',
                   torch_version=torch.__version__, transformers_version=transformers.__version__,
                   layer_forward_signature=str(inspect.signature(type(layer).forward)),
                   synthetic_config=dict(num_frames=frames, image_size=image, patch_size=patch,
                                         hidden_size=dim, num_spatial_tokens=4,
                                         attention_type='divided_space_time'),
                   cases=results, training_performed=False, validation_evaluated=False,
                   official_test_evaluated=False, pretrained_weights_loaded=False,
                   limitations=['Tiny randomly initialized model, not a trained AdaST checkpoint',
                                'A passing reshape does not establish correct spatial/temporal token semantics',
                                'No physical pruning implementation, accuracy, FLOPs or speedup demonstrated'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for r in results:
        print(f"{r['case']}: {r['status']} | input={r['input_shape']} | " +
              (f"output={r['output_shape']}" if r['status']=='PASS' else
               f"{r['error_type']}: {r['error']}"), flush=True)
    print('H29-B RESULT:', json.dumps(summary), flush=True)
    print('CLEAN H29-B COMPLETE:', out.resolve(), flush=True)

if __name__ == '__main__':
    main()
