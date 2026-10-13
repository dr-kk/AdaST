"""Clean HMDB51 split-1 TimeSformer baseline: select on held-out TRAIN validation only.
Official test split is NEVER opened by this training script.
"""
import argparse, csv, hashlib, json, os, random, time
from collections import defaultdict
from pathlib import Path
import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, TimesformerForVideoClassification

MODEL = 'facebook/timesformer-base-finetuned-k400'

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_hmdb_split(data_root, split_root, split):
    dr, sr = Path(data_root), Path(split_root)
    files = sorted(sr.glob(f'*_test_split{split}.txt'))
    if len(files) != 51: raise RuntimeError(f'Expected 51 split files, found {len(files)} in {sr}')
    classes = sorted(f.name.replace(f'_test_split{split}.txt', '') for f in files)
    c2i = {c:i for i,c in enumerate(classes)}
    train, test = [], []
    for f in files:
        cls = f.name.replace(f'_test_split{split}.txt', '')
        for line in f.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            name, flag = line.rsplit(maxsplit=1); flag=int(flag)
            if flag not in (1,2): continue
            direct, nested = dr/cls/name, dr/cls/cls/name
            if direct.is_file(): path=direct
            elif nested.is_file(): path=nested
            else:
                matches=[]
                for folder in (dr/cls, dr/cls/cls):
                    if folder.exists(): matches.extend(sorted(folder.glob(Path(name).stem+'.*')))
                if len(matches)!=1: raise FileNotFoundError(f'Expected one clip for {cls}/{name}, got {matches}')
                path=matches[0]
            (train if flag==1 else test).append((str(path.resolve()),c2i[cls]))
    return train,test,classes

def stratified_split(items, val_fraction, seed):
    groups=defaultdict(list)
    for path,label in items: groups[label].append((path,label))
    rng=random.Random(seed); tr=[]; val=[]
    for label in sorted(groups):
        group=sorted(groups[label]); rng.shuffle(group)
        n_val=max(1,round(len(group)*val_fraction))
        if n_val>=len(group): raise ValueError(f'Class {label} has insufficient training examples')
        val.extend(group[:n_val]); tr.extend(group[n_val:])
    rng.shuffle(tr); rng.shuffle(val)
    assert not set(p for p,_ in tr)&set(p for p,_ in val)
    return tr,val

def decode_video(path,nframes,train):
    container=av.open(path,metadata_errors='ignore'); frames=[]
    try:
        for frame in container.decode(video=0): frames.append(frame.to_ndarray(format='rgb24'))
    finally: container.close()
    if not frames: raise RuntimeError(f'No video frames: {path}')
    if train and len(frames)>nframes:
        edges=np.linspace(0,len(frames),nframes+1,dtype=int)
        indices=[min(random.randrange(edges[i],max(edges[i]+1,edges[i+1])),len(frames)-1) for i in range(nframes)]
    else: indices=np.linspace(0,len(frames)-1,nframes).round().astype(int).tolist()
    return [frames[i] for i in indices]

class HMDB(Dataset):
    def __init__(self,items,processor,nframes,train):
        self.items=items;self.processor=processor;self.nframes=nframes;self.train=train
    def __len__(self): return len(self.items)
    def __getitem__(self,index):
        path,label=self.items[index]
        x=self.processor(decode_video(path,self.nframes,self.train),return_tensors='pt')['pixel_values'].squeeze(0)
        return x,torch.tensor(label,dtype=torch.long),path

@torch.no_grad()
def evaluate(model,loader,device,amp):
    model.eval();n=correct1=correct5=0;total_loss=0.
    for x,y,_ in loader:
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        with torch.amp.autocast('cuda',enabled=amp and device.type=='cuda'):
            logits=model(pixel_values=x).logits;loss=F.cross_entropy(logits,y)
        count=y.numel();n+=count;total_loss+=loss.item()*count
        correct1+=(logits.argmax(1)==y).sum().item()
        correct5+=(logits.topk(min(5,logits.shape[1]),1).indices==y[:,None]).any(1).sum().item()
    return {'n':n,'loss':total_loss/n,'top1':100*correct1/n,'top5':100*correct5/n}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--split-root',required=True)
    p.add_argument('--split',type=int,default=1,choices=(1,2,3))
    p.add_argument('--num-frames',type=int,default=8);p.add_argument('--val-fraction',type=float,default=.10)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accum-steps',type=int,default=4);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--weight-decay',type=float,default=.05)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--amp',action='store_true')
    p.add_argument('--output',default='runs/hmdb51_clean_h1_dense_s1')
    a=p.parse_args()
    if not 0<a.val_fraction<.5 or min(a.epochs,a.batch_size,a.accum_steps,a.num_frames)<1: p.error('Invalid training or validation settings')
    output=Path(a.output).resolve()
    if output.exists() and any(output.iterdir()):
        p.error(f'Output folder is not empty: {output}. Use a NEW output folder; existing runs are never overwritten.')
    output.mkdir(parents=True,exist_ok=True)
    seed_all(a.seed)
    official_train,official_test,classes=read_hmdb_split(a.data_root,a.split_root,a.split)
    tr_items,val_items=stratified_split(official_train,a.val_fraction,a.seed)
    train_paths={p for p,_ in tr_items};val_paths={p for p,_ in val_items};test_paths={p for p,_ in official_test}
    if (train_paths|val_paths)&test_paths: raise RuntimeError('Official test overlap detected!')
    manifest={'protocol':'Official test held out; best checkpoint selected by validation Top-1 only',
              'split':a.split,'seed':a.seed,'val_fraction':a.val_fraction,'classes':classes,
              'train':tr_items,'val':val_items,'official_test_n_not_evaluated':len(official_test),
              'args':vars(a)}
    manifest_text=json.dumps(manifest,indent=2)
    (output/'split_manifest.json').write_text(manifest_text,encoding='utf-8')
    digest=hashlib.sha256(manifest_text.encode()).hexdigest()
    print(f'HMDB51 split {a.split}: train={len(tr_items)} val={len(val_items)} official_test={len(official_test)} (NOT EVALUATED)',flush=True)
    print(f'Manifest SHA256: {digest}',flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device,torch.cuda.get_device_name(0) if device.type=='cuda' else '',flush=True)
    proc=AutoImageProcessor.from_pretrained(MODEL)
    tr=DataLoader(HMDB(tr_items,proc,a.num_frames,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=='cuda')
    vl=DataLoader(HMDB(val_items,proc,a.num_frames,False),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda')
    model=TimesformerForVideoClassification.from_pretrained(MODEL,num_frames=a.num_frames,num_labels=51,ignore_mismatched_sizes=True).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=a.amp and device.type=='cuda')
    history=[];best=-1.;best_epoch=None
    for epoch in range(1,a.epochs+1):
        model.train();opt.zero_grad(set_to_none=True);correct=seen=0;total_loss=0.;start=time.time()
        for step,(x,y,_) in enumerate(tr,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            # Divide by actual accumulation group size, including final partial group.
            group_start=((step-1)//a.accum_steps)*a.accum_steps+1
            group_size=min(a.accum_steps,len(tr)-group_start+1)
            with torch.amp.autocast('cuda',enabled=a.amp and device.type=='cuda'):
                logits=model(pixel_values=x).logits
                raw_loss=F.cross_entropy(logits,y)
                loss=raw_loss/group_size
            scaler.scale(loss).backward()
            if step%a.accum_steps==0 or step==len(tr):
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            count=y.numel();seen+=count;correct+=(logits.argmax(1)==y).sum().item();total_loss+=raw_loss.item()*count
            if step%200==0 or step==len(tr):
                print(f'E{epoch} step {step}/{len(tr)} train_top1={100*correct/seen:.2f}% loss={total_loss/seen:.4f}',flush=True)
        val=evaluate(model,vl,device,a.amp)
        row={'epoch':epoch,'train_n':seen,'train_top1':100*correct/seen,'train_loss':total_loss/seen,
             'val_n':val['n'],'val_top1':val['top1'],'val_top5':val['top5'],'val_loss':val['loss'],
             'seconds':time.time()-start}
        history.append(row)
        with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        print(f'E{epoch}: train={row["train_top1"]:.2f}% val_top1={val["top1"]:.2f}% val_top5={val["top5"]:.2f}% time={row["seconds"]:.1f}s',flush=True)
        if val['top1']>best:
            best=val['top1'];best_epoch=epoch
            torch.save({'model':model.state_dict(),'classes':classes,'split':a.split,'num_frames':a.num_frames,
                        'metrics':val,'selected_epoch':epoch,'selection_metric':'val_top1',
                        'manifest_sha256':digest,'protocol':'Validation-selected; official test not evaluated'},output/'best_val.pt')
            print('Saved validation-selected checkpoint:',output/'best_val.pt',flush=True)
        summary={'experiment':'Clean HMDB51 H1 dense baseline','protocol':'Official test NEVER evaluated during training/selection',
                 'split':a.split,'train_n':len(tr_items),'val_n':len(val_items),
                 'official_test_n_not_evaluated':len(official_test),'best_epoch':best_epoch,
                 'best_val_top1':best,'manifest_sha256':digest,'history':history,'args':vars(a)}
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('CLEAN H1 COMPLETE:',json.dumps({'best_epoch':best_epoch,'best_val_top1':best,'checkpoint':str(output/'best_val.pt'),'test_evaluated':False}),flush=True)

if __name__=='__main__': main()
