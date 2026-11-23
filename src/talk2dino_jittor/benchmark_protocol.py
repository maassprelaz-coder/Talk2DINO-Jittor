"""Framework-neutral VOC sliding-window geometry and label conventions."""
import cv2
import numpy as np
from .preprocessing import _linear_axis_weights,IMAGENET_MEAN,IMAGENET_STD

VOC_CLASSES=("aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
             "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor")
TEMPLATES=("itap of a {}.","a bad photo of a {}.","a origami {}.","a photo of the large {}.",
           "a {} in a video game.","art of the {}.","a photo of the small {}.")
PROTOCOLS={"voc20":{"clean":False,"background":False,"classes":20},
           "voc21":{"clean":True,"background":True,"classes":21},
           "voc21_no_clean":{"clean":False,"background":True,"classes":21}}

def resize_outer(image):
    height,width=image.shape[:2]
    scale=min(2048/max(height,width),448/min(height,width))
    return cv2.resize(image,(int(width*scale+.5),int(height*scale+.5)),interpolation=cv2.INTER_LINEAR)

def windows(height,width,crop=448,stride=224):
    rows=max(height-crop+stride-1,0)//stride+1
    cols=max(width-crop+stride-1,0)//stride+1
    for row in range(rows):
        for col in range(cols):
            y1=row*stride;x1=col*stride
            y2=min(y1+crop,height);x2=min(x1+crop,width)
            y1=max(y2-crop,0);x1=max(x2-crop,0)
            yield y1,y2,x1,x2

def normalize_crop(rgb,size=448):
    pixels=rgb.astype(np.float32);height,width=pixels.shape[:2]
    if width!=size:
        pixels=np.einsum("wj,hjc->hwc",_linear_axis_weights(width,size),pixels,optimize=True)
    if height!=size:
        pixels=np.einsum("vh,hwc->vwc",_linear_axis_weights(height,size),pixels,optimize=True)
    return (pixels.transpose(2,0,1)/np.float32(255)-IMAGENET_MEAN)/IMAGENET_STD

def voc_target(mask,background):
    if background:return mask
    result=mask.astype(np.int32)-1
    result[(mask==0)|(mask==255)]=255
    return result
