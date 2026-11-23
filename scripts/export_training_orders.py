"""Export PyTorch DataLoader order once for identical Jittor/Torch sampling."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from range_download import atomic_json,sha256
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
p.add_argument("--captions",type=int,default=414113);p.add_argument("--epochs",type=int,default=100)
p.add_argument("--seed",type=int,default=123)
a=p.parse_args()
folder=a.root/"features/sampling";folder.mkdir(parents=True,exist_ok=True)
path=folder/f"torch_seed{a.seed}_n{a.captions}_e{a.epochs}.npy"
meta=path.with_suffix(".json")
config={"seed":a.seed,"captions":a.captions,"epochs":a.epochs,"batch_size":128,"drop_last":False,
        "source":"PyTorch DataLoader shuffle=True; validation iterator base-seed consumed after each epoch",
        "torch":torch.__version__}
if path.exists():
    record=json.loads(meta.read_text())
    if record["config"]!=config or not record["complete"]:raise RuntimeError("Existing orders not compatible")
    print(str(path));raise SystemExit
temporary=path.with_suffix(".npy.tmp")
orders=np.lib.format.open_memmap(temporary,mode="w+",dtype=np.uint32,shape=(a.epochs,a.captions))
torch.manual_seed(a.seed)
train=DataLoader(range(a.captions),shuffle=True,batch_size=128,num_workers=0)
val=DataLoader(range(1),shuffle=False,batch_size=128,num_workers=0)
for epoch in range(a.epochs):
    offset=0
    for batch in train:
        orders[epoch,offset:offset+len(batch)]=batch.numpy();offset+=len(batch)
    for _ in val:pass
    if offset!=a.captions:raise RuntimeError("Missing sampled captions")
orders.flush();del orders;temporary.replace(path)
atomic_json(meta,{"config":config,"complete":True,"sha256":sha256(path)})
print(str(path))
