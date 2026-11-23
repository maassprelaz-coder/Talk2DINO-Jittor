"""Export seeded official initialization for both frameworks."""
import argparse
import ast
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--seed",type=int,default=123)
a=p.parse_args(); root=a.root.resolve()
source=root/"upstream/Talk2DINO/src/model.py"
tree=ast.parse(source.read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ProjectionLayer")
namespace={"torch":torch,"nn":nn,"F":F,"yaml":yaml}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
torch.manual_seed(a.seed)
model=namespace["ProjectionLayer"](act=nn.Tanh(),hidden_layer=True,dino_embed_dim=768)
out=root/"checkpoints"/f"mapper_init_seed{a.seed}.npz"
np.savez(out,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
print(out)
