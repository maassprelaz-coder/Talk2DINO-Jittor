"""Talk2DINO sigmoid scores and the official raw-attention background cleaning."""
import jittor as jt
from jittor import nn
from .projection import l2_normalize


def background_clean(mask,patches,attention_logits,text_embedding,lambda_bg=0.2):
    """Port of DINOText.similarity_assignment_weighted (single image evaluation).

    The pinned reference weights RAW attention logits when forming each head
    embedding. Its later spatial softmax is preserved exactly.
    """
    batch,classes,height,width=mask.shape
    if batch!=1:
        raise ValueError("Pinned official background cleaning supports batch=1")
    count=height*width
    head_embedding=(attention_logits.unsqueeze(-1)*patches.unsqueeze(1)).mean(dim=2)
    head_embedding=head_embedding/jt.sqrt((head_embedding*head_embedding).sum(dim=-1,keepdims=True))
    head_similarity=text_embedding.unsqueeze(0) @ head_embedding.transpose(1,2)
    head_similarity=nn.softmax(head_similarity,dim=-1)
    attention=(attention_logits.unsqueeze(1)*head_similarity.unsqueeze(-1)).sum(dim=2)
    attention=attention/head_similarity.sum(dim=-1).unsqueeze(-1)
    attention=nn.softmax(attention,dim=-1)
    minimum=float(attention.min().item()); maximum=float(attention.max().item())
    denominator=max(maximum,maximum-minimum)
    attention=(attention-minimum)/denominator
    attention=attention*(mask.max()-mask.min())+mask.min()
    return (mask.reshape(classes,count)+lambda_bg*attention).reshape(batch,classes,height,width)/(1+lambda_bg)


def score_patches(features,text_embedding,grid_size=32,clean=True,lambda_bg=0.2):
    patches=features["patches"]
    scores=(l2_normalize(patches) @ text_embedding.transpose(1,0)).transpose(1,2)
    mask=jt.sigmoid(scores).reshape(patches.shape[0],text_embedding.shape[0],grid_size,grid_size)
    if clean:
        mask=background_clean(mask,patches,features["attention_logits"],text_embedding,lambda_bg)
    return mask


def resize_scores(mask,height,width):
    return nn.interpolate(mask,size=(height,width),mode="bilinear",align_corners=True)
