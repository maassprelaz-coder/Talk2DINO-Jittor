"""Portable optimizer-state interfaces for the formal mapper trainer."""
import ast
import numpy as np

class JittorTrainer:
    def __init__(self,initial,alignment):
        import jittor as jt
        from talk2dino_jittor.projection import ProjectionLayer,infonce_loss
        from talk2dino_jittor.optim import TorchCompatibleAdam
        self.lib=jt;self.loss_fn=infonce_loss
        jt.flags.use_cuda=1;jt.set_global_seed(123)
        self.model=ProjectionLayer(alignment_strategy=alignment)
        with np.load(initial) as weights:self.model.load_parameters({k:jt.array(weights[k]) for k,_ in self.model.named_parameters()})
        self.optimizer=TorchCompatibleAdam(self.model.parameters(),lr=1e-4)
    def loss(self,visual,text,train):
        jt=self.lib
        def calculate():return self.loss_fn(self.model(jt.array(np.asarray(visual)),jt.array(np.asarray(text))))
        if train:
            loss=calculate();value=float(loss.item());self.optimizer.step(loss);jt.sync_all();return value
        with jt.no_grad():return float(calculate().item())
    def weights(self):return {k:v.numpy() for k,v in self.model.named_parameters()}
    def state(self):
        params=list(self.model.named_parameters());g=self.optimizer.param_groups[0]
        return ({k:m.numpy() for (k,_),m in zip(params,g["m"])},
                {k:v.numpy() for (k,_),v in zip(params,g["values"])},self.optimizer.n_step)
    def restore(self,archive,step):
        jt=self.lib
        self.model.load_parameters({k:jt.array(archive["model/"+k]) for k,_ in self.model.named_parameters()})
        group=self.optimizer.param_groups[0]
        for (name,_),m,v in zip(self.model.named_parameters(),group["m"],group["values"]):
            m.update(jt.array(archive["first/"+name]));v.update(jt.array(archive["second/"+name]))
        self.optimizer.n_step=step

class TorchTrainer:
    def __init__(self,initial,alignment,root):
        import torch
        from torch import nn
        from torch.nn import functional as F
        self.lib=torch;self.alignment=alignment
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ns={"torch":torch,"nn":nn,"F":F,"np":np,"CLIPLastLayer":type("UnusedCLIPLastLayer",(),{})}
        for file,names in [("model.py",{"ProjectionLayer"}),("loss.py",{"Contrastive","ContrastiveLoss"})]:
            source=root/"upstream/Talk2DINO/src"/file
            nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name in names]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),ns)
        self.model=ns["ProjectionLayer"](act=nn.Tanh(),hidden_layer=True,dino_embed_dim=768).float().cuda()
        with np.load(initial) as weights:
            self.model.load_state_dict({k:torch.from_numpy(weights[k].copy()).cuda() for k in self.model.state_dict()})
        self.criterion=ns["ContrastiveLoss"](self.model,ltype="infonce").cuda()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4)
    def loss(self,visual,text,train):
        t=self.lib
        if self.alignment=="mean_heads" and visual.ndim==3:visual=visual.mean(1)
        images=t.from_numpy(np.array(visual,copy=True)).cuda()
        captions=t.from_numpy(np.array(text,copy=True)).cuda()
        if train:
            self.optimizer.zero_grad();self.criterion.logit_scale.grad=None
            loss=self.criterion(images,captions);value=float(loss.item())
            loss.backward();self.optimizer.step();t.cuda.synchronize();return value
        with t.no_grad():return float(self.criterion(images,captions).item())
    def weights(self):return {k:v.detach().cpu().numpy() for k,v in self.model.named_parameters()}
    def state(self):
        first={};second={};steps=[]
        for name,p in self.model.named_parameters():
            s=self.optimizer.state[p]
            first[name]=s["exp_avg"].cpu().numpy();second[name]=s["exp_avg_sq"].cpu().numpy();steps.append(int(s["step"].item()))
        if len(set(steps))!=1:raise RuntimeError("Inconsistent Adam steps")
        return first,second,steps[0]
    def restore(self,archive,step):
        t=self.lib
        self.model.load_state_dict({k:t.from_numpy(archive["model/"+k].copy()).cuda() for k in self.model.state_dict()})
        for name,p in self.model.named_parameters():
            self.optimizer.state[p]={"step":t.tensor(float(step)),
                "exp_avg":t.from_numpy(archive["first/"+name].copy()).cuda(),
                "exp_avg_sq":t.from_numpy(archive["second/"+name].copy()).cuda()}
