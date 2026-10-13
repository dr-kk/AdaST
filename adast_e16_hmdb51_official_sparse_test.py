#!/usr/bin/env python3
"""H37 validation-only trained H13 dense vs physically compact indexed sparse prototype.
No training or official test; original checkpoint/model files are never modified.
Sparse path is Python-grouped and NOT optimized; end-to-end timing includes controller.
"""
import argparse, csv, hashlib, json, math, statistics, sys, time, types
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from adast_e5_adaptive_consolidation import AdaSTE5, MODEL_NAME
from adast_hmdb51_clean_h1_dense import HMDB, read_hmdb_split
from adast_hmdb51_clean_h32_sparse_indexed_layer import indexed

EXPECTED='8b45fc6c63ea8602becd674f59f1f8892174ce352c46fe45a7745423f8405bec'

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--split-root',required=True)
    p.add_argument('--h13-checkpoint',default=r'runs\hmdb51_clean_h13_balanced_sampling_s1\best_val.pt')
    p.add_argument('--manifest',default=r'runs\hmdb51_clean_h1_dense_s1\split_manifest.json')
    p.add_argument('--output',default=r'runs\e16_hmdb51_official_sparse_test_s1')
    p.add_argument('--max-videos',type=int,default=1530,help='1530 = full official test; smaller values are PILOT ONLY')
    p.add_argument('--progress-every',type=int,default=10)
    p.add_argument('--amp',action='store_true',help='FP16 autocast; default FP32, matching H36-B')
    a=p.parse_args()
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Output folder not empty; choose a new output')
    if not 1<=a.max_videos<=1530:p.error('--max-videos must be in 1..1530')
    mp=Path(a.manifest);m=json.loads(mp.read_text(encoding='utf-8'))
    canonical=hashlib.sha256(json.dumps(m,indent=2,ensure_ascii=False).encode('utf-8')).hexdigest()
    if canonical!=EXPECTED:raise RuntimeError('Canonical manifest hash mismatch; STOP')
    tr=[(str(path),int(y)) for path,y in m['train']]
    val=[(str(path),int(y)) for path,y in m['val']]
    official_train,official_test,classes=read_hmdb_split(a.data_root,a.split_root,1)
    # Relocation-safe sample identity:
    # the canonical manifest was created under an older absolute dataset root.
    # Compare class-directory + filename + label, not drive/root path.
    def key(z):
        pth=Path(z[0])
        return (pth.parent.name.casefold(),pth.name.casefold(),int(z[1]))
    trset,valset,officialset,testset=(set(map(key,z)) for z in (tr,val,official_train,official_test))
    if (len(tr),len(val),len(official_test))!=(3213,357,1530) or len(trset)!=len(tr) or len(valset)!=len(val) or trset&valset or trset|valset!=officialset or (trset|valset)&testset or m['classes']!=classes or m['split']!=1:
        raise RuntimeError('Official split/manifest validation failed; STOP')
    if any(not Path(path).is_file() for path,_ in official_test):raise RuntimeError('Missing official-test video')
    ck=torch.load(a.h13_checkpoint,map_location='cpu',weights_only=False)
    if ck.get('manifest_sha256')!=EXPECTED or ck.get('classes')!=classes or ck.get('split')!=1 or ck.get('num_frames')!=8 or ck.get('protocol')!='Clean H13 balanced sampling; official test not evaluated':
        raise RuntimeError('H13 checkpoint metadata mismatch; STOP')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc=AutoImageProcessor.from_pretrained(MODEL_NAME)
    bb=TimesformerForVideoClassification.from_pretrained(MODEL_NAME,num_frames=8,num_labels=51,ignore_mismatched_sizes=True)
    model=AdaSTE5(bb,block=6,T=8)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.to(device).eval()
    loader=DataLoader(HMDB(official_test[:a.max_videos],proc,8,False),batch_size=1,shuffle=False,num_workers=0)
    layers=model.layers(); original={i:layers[i].forward for i in range(6,len(layers))}
    capture={}
    # This hook is registered AFTER AdaSTE5's policy hook and observes the actual consolidated tensor.
    def observe(module,args,kwargs):
        h=args[0] if args else kwargs['hidden_states']
        capture['full']=h
        return None
    handle=layers[6].register_forward_pre_hook(observe,with_kwargs=True)
    out.mkdir(parents=True,exist_ok=False)
    rows=[]; errors=[]
    fields=['index','label','dense_prediction','sparse_prediction','dense_correct','sparse_correct','dense_top5','sparse_top5','k_per_frame','selected_frames','original_patch_tokens','retained_patch_tokens','original_sequence_length','compact_sequence_length','dense_ms','sparse_ms','dense_peak_gpu_mb','sparse_peak_gpu_mb']
    csvpath=out/'per_video.csv'
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        try:
            with torch.inference_mode():
                for ix,(x,y,_) in enumerate(loader):
                    x=x.to(device,non_blocking=True);target=int(y.item())
                    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                    sync(device);start=time.perf_counter()
                    with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                        dense=model(x).logits.detach().float()
                    sync(device);dense_ms=(time.perf_counter()-start)*1000
                    dense_peak=(torch.cuda.max_memory_allocated()/1048576 if device.type=='cuda' else None)
                    full=capture['full'];frames=8;original_patches=full.shape[1]-1
                    if full.shape[0]!=1 or original_patches%frames:raise RuntimeError('Unexpected block6 shape')
                    mask=full[:,1:,:].abs().amax(-1)[0]>0
                    positions=torch.nonzero(mask,as_tuple=True)[0]
                    k=int(model.last_k[0]);s=int(model.last_s[0]);retained=int(positions.numel())
                    if retained!=k*s or not 0<retained<original_patches:raise RuntimeError(f'Unsafe sparse mask at official-test index {ix}: {retained} vs {k*s}')
                    ids=torch.stack([positions%frames,positions//frames],dim=1).cpu()
                    compact=torch.cat([full[:,:1,:],full[:,1:,:][:,positions,:]],dim=1)
                    # Recompute controller for sparse run so its cost is INCLUDED in end-to-end latency.
                    # After the original policy hook, replace its full-length tensor with compact tensor.
                    def replace(module,args,kwargs):
                        h=args[0] if args else kwargs['hidden_states']
                        if h.shape!=full.shape:raise RuntimeError('Controller changed sequence geometry')
                        new_mask=h[:,1:,:].abs().amax(-1)[0]>0
                        if not torch.equal(new_mask,mask):raise RuntimeError('Controller decision changed between matched runs')
                        z=torch.cat([h[:,:1,:],h[:,1:,:][:,positions,:]],dim=1)
                        if args:return (z,)+args[1:],kwargs
                        kw=dict(kwargs);kw['hidden_states']=z;return args,kw
                    handle.remove();handle=None
                    replace_handle=layers[6].register_forward_pre_hook(replace,with_kwargs=True)
                    try:
                        for li in range(6,len(layers)):
                            def make_sparse(layer):
                                def forward(self,hidden_states,output_attentions=False):
                                    if output_attentions:raise RuntimeError('Attention output unsupported')
                                    if hidden_states.shape[1]!=retained+1:raise RuntimeError('Compact sequence changed unexpectedly')
                                    return (indexed(self,hidden_states,ids,frames),)
                                return types.MethodType(forward,layer)
                            layers[li].forward=make_sparse(layers[li])
                        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                        sync(device);start=time.perf_counter()
                        with torch.autocast('cuda',enabled=a.amp and device.type=='cuda',dtype=torch.float16):
                            sparse=model(x).logits.detach().float()
                        sync(device);sparse_ms=(time.perf_counter()-start)*1000
                        sparse_peak=(torch.cuda.max_memory_allocated()/1048576 if device.type=='cuda' else None)
                    finally:
                        replace_handle.remove()
                        for li,fn in original.items():layers[li].forward=fn
                        handle=layers[6].register_forward_pre_hook(observe,with_kwargs=True)
                    if not torch.isfinite(sparse).all():raise RuntimeError(f'Nonfinite sparse logits at official-test index {ix}')
                    dp=int(dense.argmax(1));sp=int(sparse.argmax(1))
                    row=dict(index=ix,label=target,dense_prediction=dp,sparse_prediction=sp,dense_correct=int(dp==target),sparse_correct=int(sp==target),dense_top5=int(target in dense.topk(5,dim=1).indices[0].tolist()),sparse_top5=int(target in sparse.topk(5,dim=1).indices[0].tolist()),k_per_frame=k,selected_frames=s,original_patch_tokens=original_patches,retained_patch_tokens=retained,original_sequence_length=full.shape[1],compact_sequence_length=retained+1,dense_ms=round(dense_ms,4),sparse_ms=round(sparse_ms,4),dense_peak_gpu_mb=round(dense_peak,3) if dense_peak is not None else '',sparse_peak_gpu_mb=round(sparse_peak,3) if sparse_peak is not None else '')
                    writer.writerow(row);f.flush();rows.append(row)
                    if (ix+1)%a.progress_every==0 or ix+1==a.max_videos:
                        print(f'E16 PROGRESS: {ix+1}/{a.max_videos} dense_top1={100*sum(r["dense_correct"] for r in rows)/len(rows):.2f}% sparse_top1={100*sum(r["sparse_correct"] for r in rows)/len(rows):.2f}% dense_ms={statistics.median(r["dense_ms"] for r in rows):.1f} sparse_ms={statistics.median(r["sparse_ms"] for r in rows):.1f}',flush=True)
        finally:
            if handle is not None:handle.remove()
            for li,fn in original.items():layers[li].forward=fn
    def mean(field):return sum(r[field] for r in rows)/len(rows)
    def median(field):return statistics.median(r[field] for r in rows)
    result=dict(experiment='E16 frozen H13 dense vs indexed physically compact sparse official HMDB51 Split-1 test',status='PASS' if len(rows)==a.max_videos else 'INCOMPLETE',full_official_test=(len(rows)==1530),n=len(rows),official_test_n=1530,manifest_canonical_sha256=canonical,checkpoint_loaded_strictly=True,device=str(device),amp=a.amp,dense_top1_percent=100*mean('dense_correct'),sparse_top1_percent=100*mean('sparse_correct'),dense_top5_percent=100*mean('dense_top5'),sparse_top5_percent=100*mean('sparse_top5'),mean_retained_patch_tokens=mean('retained_patch_tokens'),mean_retained_fraction=mean('retained_patch_tokens')/rows[0]['original_patch_tokens'],original_sequence_length=rows[0]['original_sequence_length'],mean_compact_sequence_length=mean('compact_sequence_length'),dense_latency_median_ms=median('dense_ms'),sparse_latency_median_ms=median('sparse_ms'),dense_latency_mean_ms=mean('dense_ms'),sparse_latency_mean_ms=mean('sparse_ms'),dense_peak_gpu_mb_median=median('dense_peak_gpu_mb') if device.type=='cuda' else None,sparse_peak_gpu_mb_median=median('sparse_peak_gpu_mb') if device.type=='cuda' else None,latency_includes_controller_and_consolidation=True,latency_includes_video_decode=False,latency_includes_cpu_to_gpu_transfer=False,matched_order='dense then sparse; warmup not excluded',training_performed=False,official_test_evaluated=(len(rows)==1530),limitations=['Python-grouped indexed sparse prototype is not optimized for speed.','Fixed dense-then-sparse order and no warmup exclusion can bias latency; repeat with balanced order before a publication-grade speedup claim.','Peak GPU allocation measured separately for each pass but model weights remain resident.','Official HMDB51 Split-1 test; frozen H13 checkpoint; no test-set tuning or checkpoint reselection.','Changing attention context can alter accuracy; sparse path is not the original trained dense method.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('E16 RESULT:',json.dumps(result),flush=True)
    print('E16 COMPLETE:',out,flush=True)

if __name__=='__main__':main()


