"""
AdaST HMDB51 Cross-Dataset Validation — H1 Dense Baseline
Protocol: official HMDB51 split files, selectable split 1/2/3.
Expected:
  DATA_ROOT/
    brush_hair/*.avi
    cartwheel/*.avi
    ...
  SPLIT_ROOT/
    brush_hair_test_split1.txt
    ...
Split-file labels: 1=train, 2=test, 0=unused.

This stage fine-tunes a 51-class TimeSformer baseline. Its best checkpoint becomes
the initialization/reference for AdaST HMDB51 E5/E6 cross-dataset validation.
"""
import os, json, random, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import av

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def read_hmdb_split(data_root, split_root, split=1):
    sr=Path(split_root); dr=Path(data_root)
    files=sorted(sr.glob(f"*_test_split{split}.txt"))
    if len(files)!=51:
        raise RuntimeError(f"Expected 51 HMDB51 split-{split} files in {sr}, found {len(files)}")
    classes=sorted([f.name.replace(f"_test_split{split}.txt","") for f in files])
    c2i={c:i for i,c in enumerate(classes)}
    train=[]; test=[]
    for f in files:
        cls=f.name.replace(f"_test_split{split}.txt","")
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            name,flag=line.rsplit(maxsplit=1); flag=int(flag)
            # Support both standard HMDB51 layout:
            #   root/class/video.avi
            # and this machine's nested extraction:
            #   root/class/class/video.avi
            class_dir=dr/cls
            direct=class_dir/name
            nested=class_dir/cls/name
            if direct.exists():
                p=direct
            elif nested.exists():
                p=nested
            else:
                stem=Path(name).stem
                candidates=[]
                for d in (class_dir, class_dir/cls):
                    if d.exists():
                        candidates.extend(d.glob(stem+".*"))
                if not candidates:
                    raise FileNotFoundError(
                        f"Official split clip not found: {name}\n"
                        f"Checked: {direct}\nChecked: {nested}"
                    )
                p=candidates[0]
            item=(str(p),c2i[cls])
            if flag==1: train.append(item)
            elif flag==2: test.append(item)
    return train,test,classes

def decode_video(path, nframes=8, train=False):
    # HMDB51 AVI files may contain non-UTF8 textual metadata.
    # Ignore malformed metadata only; decoded video frames are unchanged.
    container=av.open(path, metadata_errors="ignore")
    frames=[]
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    if len(frames)==0: raise RuntimeError(f"No frames decoded: {path}")
    if train and len(frames)>nframes:
        # temporal jitter while preserving uniform coverage
        edges=np.linspace(0,len(frames),nframes+1,dtype=int)
        idx=[random.randrange(edges[i],max(edges[i]+1,edges[i+1])) for i in range(nframes)]
        idx=[min(j,len(frames)-1) for j in idx]
    else:
        idx=np.linspace(0,len(frames)-1,nframes).round().astype(int).tolist()
    return [frames[i] for i in idx]

class HMDB(Dataset):
    def __init__(self, items, processor, nframes=8, train=False):
        self.items=items; self.processor=processor; self.nframes=nframes; self.train=train
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y=self.items[i]
        fr=decode_video(p,self.nframes,self.train)
        x=self.processor(fr,return_tensors="pt")["pixel_values"].squeeze(0)
        return x,torch.tensor(y,dtype=torch.long),p

def evaluate(model,loader,device,amp):
    model.eval(); n=0;c1=0;c5=0;loss_sum=0.0
    ce=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y,_ in loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.amp.autocast("cuda",enabled=amp and device.type=="cuda"):
                o=model(pixel_values=x); logits=o.logits; loss=ce(logits,y)
            n+=y.numel();loss_sum+=loss.item()*y.numel()
            c1+=(logits.argmax(1)==y).sum().item()
            k=min(5,logits.shape[1]);c5+=(logits.topk(k,1).indices==y[:,None]).any(1).sum().item()
    return {"loss":loss_sum/n,"top1":100*c1/n,"top5":100*c5/n,"n":n}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True);p.add_argument("--split-root",required=True)
    p.add_argument("--split",type=int,default=1,choices=[1,2,3])
    p.add_argument("--num-frames",type=int,default=8)
    p.add_argument("--epochs",type=int,default=8);p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--accum-steps",type=int,default=4);p.add_argument("--workers",type=int,default=2)
    p.add_argument("--lr",type=float,default=2e-5);p.add_argument("--weight-decay",type=float,default=0.05)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--amp",action="store_true")
    p.add_argument("--output",default="runs/hmdb51_dense_s1")
    a=p.parse_args();seed_all(a.seed);os.makedirs(a.output,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_items,test_items,classes=read_hmdb_split(a.data_root,a.split_root,a.split)
    print(f"HMDB51 split {a.split}: train={len(train_items)}, test={len(test_items)}, classes={len(classes)}")
    print("Resolved sample:", train_items[0][0] if train_items else test_items[0][0])
    print("Device:",device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")
    proc=AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    tr=HMDB(train_items,proc,a.num_frames,True);te=HMDB(test_items,proc,a.num_frames,False)
    dl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True)
    tl=DataLoader(te,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True)
    model=TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",num_frames=a.num_frames,
        num_labels=51,ignore_mismatched_sizes=True).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scaler=torch.amp.GradScaler("cuda",enabled=a.amp and device.type=="cuda")
    best=-1.;hist=[]
    for ep in range(1,a.epochs+1):
        model.train();opt.zero_grad(set_to_none=True);seen=0;correct=0;t0=time.time()
        for it,(x,y,_) in enumerate(dl,1):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            with torch.amp.autocast("cuda",enabled=a.amp and device.type=="cuda"):
                logits=model(pixel_values=x).logits
                loss=nn.functional.cross_entropy(logits,y)/a.accum_steps
            scaler.scale(loss).backward()
            if it%a.accum_steps==0 or it==len(dl):
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            seen+=y.numel();correct+=(logits.argmax(1)==y).sum().item()
        r=evaluate(model,tl,device,a.amp)
        row={"epoch":ep,"train_top1":100*correct/seen,**r,"seconds":time.time()-t0};hist.append(row)
        print(f"E{ep}: train={row['train_top1']:.2f}% test={r['top1']:.2f}% top5={r['top5']:.2f}% time={row['seconds']:.1f}s")
        if r["top1"]>best:
            best=r["top1"]
            torch.save({"model":model.state_dict(),"classes":classes,"split":a.split,
                        "num_frames":a.num_frames,"metrics":r},Path(a.output)/"best.pt")
            print("Saved new best.")
    summary={"dataset":"HMDB51","split":a.split,"train_n":len(train_items),"test_n":len(test_items),
             "classes":classes,"best_top1":best,"history":hist}
    (Path(a.output)/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Summary:",Path(a.output)/"summary.json")

if __name__=="__main__": main()
