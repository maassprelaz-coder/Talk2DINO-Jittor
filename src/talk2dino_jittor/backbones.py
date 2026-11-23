"""Frozen DINOv2 ViT-B/14-register and CLIP ViT-B/16 text encoder in Jittor.

Architectures follow the original Meta DINOv2 and OpenAI CLIP implementations.
Weight inputs are FP32 NPZ exports from those official checkpoints.
"""
import math
import numpy as np
import jittor as jt
from jittor import nn
from .runtime import configure_fp32
configure_fp32()
from .interpolation import resize_position_embedding


def load_frozen_weights(path):
    with np.load(path) as archive:
        return {name:jt.array(archive[name]).float32().stop_grad() for name in archive.files}


def linear(x, weight, bias=None):
    result=x @ weight.transpose(1,0)
    return result if bias is None else result+bias


def layer_norm(x,weight,bias,epsilon):
    mean=x.mean(dim=-1,keepdims=True)
    centered=x-mean
    variance=(centered*centered).mean(dim=-1,keepdims=True)
    return centered/jt.sqrt(variance+epsilon)*weight+bias


def attention(x,weights,prefix,heads,causal_mask=None):
    batch,length,width=x.shape
    if prefix.endswith("attn."):
        if prefix+"qkv.weight" in weights:
            qkv=linear(x,weights[prefix+"qkv.weight"],weights[prefix+"qkv.bias"])
            output_weight=weights[prefix+"proj.weight"]; output_bias=weights[prefix+"proj.bias"]
        else:
            qkv=linear(x,weights[prefix+"in_proj_weight"],weights[prefix+"in_proj_bias"])
            output_weight=weights[prefix+"out_proj.weight"]; output_bias=weights[prefix+"out_proj.bias"]
    else:
        raise ValueError("Expected attention parameter prefix")
    qkv=qkv.reshape(batch,length,3,heads,width//heads).permute(2,0,3,1,4)
    q,k,v=qkv[0],qkv[1],qkv[2]
    logits=(q/math.sqrt(width//heads)) @ k.transpose(-2,-1)
    if causal_mask is not None: logits=logits+causal_mask
    result=nn.softmax(logits,dim=-1) @ v
    result=result.transpose(1,2).reshape(batch,length,width)
    return linear(result,output_weight,output_bias),logits


class DinoV2ViTB14Registers(nn.Module):
    def __init__(self,weight_path):
        self.w=load_frozen_weights(weight_path)
        self.original_position=self.w["pos_embed"].numpy()
        self.position_cache={}

    def _position(self,height,width):
        key=(height,width)
        if key not in self.position_cache:
            array=resize_position_embedding(self.original_position,height,width)
            self.position_cache[key]=jt.array(array).stop_grad()
        return self.position_cache[key]

    def execute(self,image):
        batch,channels,height,width=image.shape
        if channels!=3 or height%14 or width%14:
            raise ValueError("Expected RGB image with dimensions divisible by patch size 14")
        gh,gw=height//14,width//14
        # Conv2d(kernel=stride=14) expressed as patch extraction + matrix product.
        patches=image.reshape(batch,3,gh,14,gw,14).permute(0,2,4,1,3,5).reshape(batch,gh*gw,3*14*14)
        patches=linear(patches,self.w["patch_embed.proj.weight"].reshape(768,-1),
                       self.w["patch_embed.proj.bias"])
        cls=self.w["cls_token"].broadcast((batch,1,768))
        x=jt.concat((cls,patches),dim=1)+self._position(gh,gw)
        registers=self.w["register_tokens"].broadcast((batch,4,768))
        x=jt.concat((x[:,:1],registers,x[:,1:]),dim=1)
        last_logits=None
        for index in range(12):
            prefix=f"blocks.{index}."
            normed=layer_norm(x,self.w[prefix+"norm1.weight"],self.w[prefix+"norm1.bias"],1e-6)
            attended,logits=attention(normed,self.w,prefix+"attn.",12)
            x=x+attended*self.w[prefix+"ls1.gamma"]
            normed=layer_norm(x,self.w[prefix+"norm2.weight"],self.w[prefix+"norm2.bias"],1e-6)
            hidden=linear(normed,self.w[prefix+"mlp.fc1.weight"],self.w[prefix+"mlp.fc1.bias"])
            hidden=hidden*0.5*(1+jt.erf(hidden/math.sqrt(2)))
            hidden=linear(hidden,self.w[prefix+"mlp.fc2.weight"],self.w[prefix+"mlp.fc2.bias"])
            x=x+hidden*self.w[prefix+"ls2.gamma"]
            if index==11: last_logits=logits[:,:,0,5:]
        x=layer_norm(x,self.w["norm.weight"],self.w["norm.bias"],1e-6)
        patches=x[:,5:]
        head_features=(patches.unsqueeze(1)*nn.softmax(last_logits,dim=-1).unsqueeze(-1)).mean(dim=2)
        average=(patches*nn.softmax(last_logits.mean(dim=1),dim=-1).unsqueeze(-1)).mean(dim=1)
        return {"patches":patches,"cls":x[:,0],"attention_logits":last_logits,
                "head_features":head_features,"average_feature":average}


class ClipViTB16Text(nn.Module):
    def __init__(self,weight_path):
        self.w=load_frozen_weights(weight_path)
        mask=np.triu(np.full((77,77),-np.inf,dtype=np.float32),1)
        self.mask=jt.array(mask).stop_grad()

    def execute(self,tokens):
        if tokens.ndim!=2 or tokens.shape[1]!=77:
            raise ValueError("CLIP tokens must have shape [batch,77]")
        x=self.w["token_embedding.weight"][tokens.int32()]+self.w["positional_embedding"]
        for index in range(12):
            prefix=f"transformer.resblocks.{index}."
            normed=layer_norm(x,self.w[prefix+"ln_1.weight"],self.w[prefix+"ln_1.bias"],1e-5)
            attended,_=attention(normed,self.w,prefix+"attn.",8,self.mask)
            x=x+attended
            normed=layer_norm(x,self.w[prefix+"ln_2.weight"],self.w[prefix+"ln_2.bias"],1e-5)
            hidden=linear(normed,self.w[prefix+"mlp.c_fc.weight"],self.w[prefix+"mlp.c_fc.bias"])
            hidden=hidden*jt.sigmoid(1.702*hidden)
            x=x+linear(hidden,self.w[prefix+"mlp.c_proj.weight"],self.w[prefix+"mlp.c_proj.bias"])
        x=layer_norm(x,self.w["ln_final.weight"],self.w["ln_final.bias"],1e-5)
        end_index=tokens.argmax(dim=-1)[0]
        return x[jt.arange(tokens.shape[0]),end_index] @ self.w["text_projection"]
