#!/usr/bin/env python3
"""H29 static feasibility audit. No dataset, checkpoint, GPU, or test access.

Run against ORIGINAL AdaSTE5 and H2 source files. This audit does not implement
pruning or assert that a shorter sequence is supported by downstream TimeSformer.
"""
import argparse, ast, hashlib, json
from pathlib import Path

def method(tree, cls, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return next((n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    return None

def snippets(src, words):
    return [{'line': i, 'text': line.strip()} for i, line in enumerate(src.splitlines(), 1)
            if any(word in line for word in words)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-source', required=True)
    p.add_argument('--h2-source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()): p.error('Choose a NEW/empty output directory')
    files = [Path(a.model_source).resolve(), Path(a.h2_source).resolve()]
    sources = [f.read_text(encoding='utf-8-sig') for f in files]
    trees = [ast.parse(src, filename=str(f)) for f, src in zip(files, sources)]
    init = method(trees[0], 'AdaSTE5', '__init__')
    hook = method(trees[0], 'AdaSTE5', '_hook')
    consolidator = method(trees[0], 'AdaptiveTokenConsolidator', 'forward')
    if not all((init, hook, consolidator)):
        raise RuntimeError('Required AdaSTE5/consolidator methods not found; verify original sources')
    hs = ast.get_source_segment(sources[0], hook)
    cs = ast.get_source_segment(sources[0], consolidator)
    checks = {
        'block6_forward_pre_hook': 'register_forward_pre_hook(self._hook' in ast.get_source_segment(sources[0], init),
        'requires_original_T_by_N_layout': 'reshape(b,self.T,n,d)' in hs or 'reshape(b, self.T, n, d)' in hs,
        'computes_joint_keep_mask': 'joint_keep=spatial_hard*temporal_gate' in hs.replace(' ', ''),
        'consolidator_uses_full_pristine_patch': 'self.consolidator(pristine_patch,joint_keep)' in hs.replace(' ', ''),
        'reassembles_original_sequence': 'torch.cat([cls,joint_patch],1)' in hs.replace(' ', ''),
        'consolidator_outputs_full_length': 'out = pristine_patch * keep_mask.unsqueeze(-1)' in cs,
        'h2_declares_no_physical_pruning': 'physical_token_deletion=False' in sources[1],
    }
    report = {
        'experiment': 'Clean H29 static physical-pruning feasibility audit',
        'source_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'checks': checks,
        'observations': {
            'current_selection': 'Joint spatial/temporal mask is passed to consolidator; the output is concatenated with CLS without compacting the sequence.',
            'insertion': 'AdaSTE5 intercepts encoder block 6 input; token selection/consolidation runs after preceding encoder blocks.',
            'critical_constraint': 'Shortening a sequence at this hook may violate downstream divided space-time attention reshape/indexing assumptions; not established by static audit.',
            'checkpoint_compatibility': 'A different physical-pruning forward path may alter predictions even with identical checkpoint weights; require separate correctness/accuracy evaluation.',
            'cost': 'Controller, selection, and consolidation overhead must be included in end-to-end timing. Existing retention numbers are not measured speedup.',
        },
        'decision': 'NO-GO for direct replacement of joint_patch by compact tokens in existing AdaSTE5 hook without a downstream TimeSformer compatibility probe.',
        'next_minimal_probe': 'Inspect installed Transformers TimeSformer attention forward for temporal/spatial reshapes and run a synthetic input shape probe (no dataset/test) before modifying architecture.',
        'training_performed': False, 'validation_evaluated': False, 'official_test_evaluated': False,
    }
    if not all(checks.values()):
        report['decision'] = 'SOURCE MISMATCH: review failed static checks before proceeding.'
    out.mkdir(parents=True, exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out/'source_locations.json').write_text(json.dumps({
        'hook': snippets(sources[0], ['register_forward_pre_hook', 'reshape(b,self.T,n,d)', 'joint_keep=', 'self.consolidator(', 'torch.cat([cls,joint_patch]', 'out = pristine_patch * keep_mask']),
        'h2': snippets(sources[1], ['physical_token_deletion=False'])}, indent=2), encoding='utf-8')
    print('H29 RESULT:', json.dumps(report), flush=True)
    print('CLEAN H29 COMPLETE:', out/'summary.json', flush=True)

if __name__ == '__main__': main()
