"""Separate Torch reference and standalone Jittor evaluation backends."""
import ast,os
import numpy as np

def official_class(root,name):
    return next(node for node in ast.parse((root/"upstream/Talk2DINO/src/model.py").read_text()).body
                if isinstance(node,ast.ClassDef) and node.name==name)

class TorchBackend:
    def __init__(self,root,weights,prompts,template_count):
        import torch,clip
        from torch import nn
        from torch.nn import functional as F
        self.lib=torch;self.F=F
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        namespace={"torch":torch,"nn":nn,"F":F}
        exec(compile(ast.Module(body=[official_class(root,"ProjectionLayer")],type_ignores=[]),"<pinned ProjectionLayer>","exec"),namespace)
        mapper=namespace["ProjectionLayer"](act=nn.Tanh(),hidden_layer=True,dino_embed_dim=768).eval().cuda()
        with np.load(weights) as archive:
            mapper.load_state_dict({k:torch.from_numpy(archive[k].copy()).cuda() for k in mapper.state_dict()})
        model,_=clip.load("ViT-B/16",device="cpu",jit=False,
            download_root=os.path.join(os.environ.get("TALK2DINO_RUNTIME", os.path.expanduser("~/.local/share/talk2dino")), "clip-cache"))
        model=model.float().eval().cuda()
        with torch.no_grad():
            tokens=clip.tokenize(prompts).cuda()
            raw=torch.cat([model.encode_text(tokens[i:i+32]) for i in range(0,len(tokens),32)])
            self.embedding=F.normalize(mapper.project_clip_txt(raw.reshape(-1,template_count,512).mean(1)),dim=-1)
        del model,mapper,raw;torch.cuda.empty_cache()
        self.model=torch.hub.load(str(root/"upstream/dinov2"),"dinov2_vitb14_reg",source="local",pretrained=True).float().eval().cuda()
        self.capture={}
        self.model.blocks[-1].attn.qkv.register_forward_hook(lambda m,i,o:self.capture.__setitem__("qkv",o))
        source=root/"upstream/Talk2DINO/src/open_vocabulary_segmentation/models/dinotext/dinotext.py"
        cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="DINOText")
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="similarity_assignment_weighted")
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),"exec"),namespace)
        self.clean_fn=namespace["similarity_assignment_weighted"]

    def crop_scores(self,array,shape,with_clean=True):
        t=self.lib;F=self.F
        with t.no_grad():
            features=self.model.forward_features(t.from_numpy(array[None]).cuda())["x_norm_patchtokens"]
            qkv=self.capture["qkv"].reshape(1,-1,3,12,64).permute(2,0,3,1,4)
            attention=((qkv[0]*.125)@qkv[1].transpose(-2,-1))[:,:,0,5:]
            raw=(F.normalize(features,dim=-1)@self.embedding.T).transpose(1,2).sigmoid().reshape(1,-1,32,32)
            if not with_clean:
                return {False:F.interpolate(raw,size=shape,mode="bilinear",align_corners=True)[0].cpu().numpy()}
            clean=self.clean_fn(None,raw,features.reshape(1,32,32,768).permute(0,3,1,2),attention,self.embedding,.2)
            return {False:F.interpolate(raw,size=shape,mode="bilinear",align_corners=True)[0].cpu().numpy(),
                    True:F.interpolate(clean,size=shape,mode="bilinear",align_corners=True)[0].cpu().numpy()}

    def restore(self,scores,shape):
        with self.lib.no_grad():
            return self.F.interpolate(self.lib.from_numpy(scores[None]).cuda(),size=shape,mode="bilinear",
                                      align_corners=False)[0].cpu().numpy()

class JittorBackend:
    def __init__(self,root,weights,prompts,template_count):
        import jittor as jt
        from talk2dino_jittor.backbones import DinoV2ViTB14Registers,ClipViTB16Text
        from talk2dino_jittor.projection import ProjectionLayer,l2_normalize
        from talk2dino_jittor.tokenizer import tokenize
        self.lib=jt;jt.flags.use_cuda=1
        mapper=ProjectionLayer()
        with np.load(weights) as archive:
            mapper.load_parameters({k:jt.array(archive[k]) for k,_ in mapper.named_parameters()})
        model=ClipViTB16Text(root/"checkpoints/clip_vitb16_text_fp32.npz")
        with jt.no_grad():
            tokens=tokenize(prompts)
            raw=jt.concat([model(jt.array(tokens[i:i+32])) for i in range(0,len(tokens),32)],dim=0)
            self.embedding=l2_normalize(mapper.project_clip_txt(raw.reshape(-1,template_count,512).mean(dim=1))).stop_grad()
            self.embedding.sync()
        del model,mapper,raw;jt.gc()
        self.model=DinoV2ViTB14Registers(root/"checkpoints/dinov2_vitb14_reg_fp32.npz")

    def crop_scores(self,array,shape,with_clean=True):
        from talk2dino_jittor.inference import score_patches,resize_scores
        jt=self.lib
        with jt.no_grad():
            features=self.model(jt.array(array[None]))
            raw=resize_scores(score_patches(features,self.embedding,clean=False),*shape)[0].numpy()
            if not with_clean:
                del features;jt.gc();return {False:raw}
            clean=resize_scores(score_patches(features,self.embedding,clean=True),*shape)[0].numpy()
            del features;jt.gc()
            return {False:raw,True:clean}

    def restore(self,scores,shape):
        jt=self.lib
        with jt.no_grad():
            return jt.nn.interpolate(jt.array(scores[None]),size=shape,mode="bilinear",align_corners=False)[0].numpy()
