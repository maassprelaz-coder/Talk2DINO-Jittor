"""Separable antialiased bicubic interpolation for frozen DINOv2 positions.

Matches the register model's align_corners=False / antialias=True convention.
This operates on frozen NumPy weight arrays; no Torch dependency at inference.
"""
import numpy as np


def _cubic(x):
    a=-0.5
    x=np.abs(x)
    return np.where(x<1,((a+2)*x-(a+3))*x*x+1,
                    np.where(x<2,((a*x-5*a)*x+8*a)*x-4*a,0))


def _axis_weights(source_size, target_size):
    scale=source_size/target_size
    kernel_scale=max(scale,1.0)
    support=2*kernel_scale
    weights=np.zeros((target_size,source_size),dtype=np.float32)
    for out in range(target_size):
        center=scale*(out+0.5)
        left=max(int(center-support+0.5),0)
        right=min(int(center+support+0.5),source_size)
        indices=np.arange(left,right)
        coeff=_cubic((indices-center+0.5)/kernel_scale)
        coeff=coeff/coeff.sum()
        weights[out,left:right]=coeff.astype(np.float32)
    return weights


def resize_position_embedding(position, height, width):
    position=np.asarray(position,dtype=np.float32)
    original=int(round((position.shape[1]-1)**0.5))
    if original*original!=position.shape[1]-1:
        raise ValueError("Expected square original positional grid")
    if height==width==original:
        return position.copy()
    patches=position[0,1:].reshape(original,original,-1)
    horizontal=_axis_weights(original,width)
    vertical=_axis_weights(original,height)
    # Interpolate horizontal first, matching PyTorch's separable AA operator.
    first=np.einsum("wj,hjc->hwc",horizontal,patches,optimize=True)
    resized=np.einsum("vh,hwc->vwc",vertical,first,optimize=True)
    return np.concatenate((position[:,:1],resized.reshape(1,height*width,-1)),axis=1)
