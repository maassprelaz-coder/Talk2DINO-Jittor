"""PIL/NumPy version of official Resize(448), CenterCrop(448), ToTensor, Normalize."""
from PIL import Image
import numpy as np

IMAGENET_MEAN=np.array([0.485,0.456,0.406],dtype=np.float32)[:,None,None]
IMAGENET_STD=np.array([0.229,0.224,0.225],dtype=np.float32)[:,None,None]

def training_image(path,size=448):
    with Image.open(path) as opened:
        image=opened.convert("RGB")
        width,height=image.size
        if width<height:
            target=(size,int(size*height/width))
        else:
            target=(int(size*width/height),size)
        image=image.resize(target,resample=Image.Resampling.BICUBIC)
        left=int(round((target[0]-size)/2.0)); top=int(round((target[1]-size)/2.0))
        image=image.crop((left,top,left+size,top+size))
        array=np.asarray(image,dtype=np.float32).transpose(2,0,1)/np.float32(255.0)
    return (array-IMAGENET_MEAN)/IMAGENET_STD


def _linear_axis_weights(source,target):
    scale=np.float32(source/target)
    stretch=max(scale,np.float32(1.0))
    inverse=np.float32(1.0)/stretch
    half=np.float32(0.5)
    weights=np.zeros((target,source),dtype=np.float32)
    for out in range(target):
        center=scale*(np.float32(out)+half)
        left=max(int(center-stretch+half),0)
        right=min(int(center+stretch+half),source)
        index=np.arange(left,right,dtype=np.float32)
        values=np.maximum(np.float32(1.0)-np.abs((index-center+half)*inverse),np.float32(0.0))
        weights[out,left:right]=values/values.sum(dtype=np.float32)
    return weights


def square_inference_image(path,size=448):
    """Match torchvision's bilinear Tensor Resize with antialias=True."""
    with Image.open(path) as opened:
        original=np.asarray(opened.convert("RGB")).copy()
    pixels=original.astype(np.float32)
    height,width=original.shape[:2]
    first=np.einsum("wj,hjc->hwc",_linear_axis_weights(width,size),pixels,optimize=True)
    resized=np.einsum("vh,hwc->vwc",_linear_axis_weights(height,size),first,optimize=True)
    array=resized.transpose(2,0,1)/np.float32(255.0)
    return (array-IMAGENET_MEAN)/IMAGENET_STD,original
