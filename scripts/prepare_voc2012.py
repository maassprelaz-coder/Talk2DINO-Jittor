"""Fetch VOC2012 validation data from a mirror and verify the official MD5."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
from PIL import Image
import numpy as np
from range_download import download,atomic_json

URL="https://huggingface.co/Peyiloo/peyiloo/resolve/0f3a1e280eac7eaa8be4d91d79a920b6a040b08c/VOCtrainval_11-May-2012.tar"
MD5="6cd6e144f989b92b3379bac3b3de84fd"
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    archive=root/"data/archives/VOCtrainval_11-May-2012.tar"
    receipt=download(URL,archive,workers=4)
    digest=hashlib.md5()
    with archive.open("rb") as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b""):digest.update(chunk)
    if digest.hexdigest()!=MD5:raise RuntimeError("VOC archive does not match torchvision's official MD5")
    base=root/"data/VOCdevkit/VOC2012"
    with tarfile.open(archive,"r:") as source:
        prefix="VOCdevkit/VOC2012/"
        names=source.extractfile(prefix+"ImageSets/Segmentation/val.txt").read().decode().splitlines()
        if not all(re.fullmatch(r"\d{4}_\d{6}",name) for name in names):
            raise RuntimeError("Unexpected VOC validation IDs")
        for relative in ["ImageSets/Segmentation/val.txt"]+[folder+"/"+name+suffix
                for name in names for folder,suffix in [("JPEGImages",".jpg"),("SegmentationClass",".png")]]:
            target=base/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(source.extractfile(prefix+relative).read())
    for name in names:
        with Image.open(base/"JPEGImages"/(name+".jpg")) as im:
            shape=im.size
        with Image.open(base/"SegmentationClass"/(name+".png")) as im:
            mask=np.asarray(im)
            if im.size!=shape:raise RuntimeError("VOC image/mask dimensions differ")
            if not set(np.unique(mask)).issubset(set(range(21))|{255}):
                raise RuntimeError("Unexpected VOC mask labels")
    metadata={"dataset":"PASCAL VOC 2012","split":"segmentation validation","images":len(names),
              "official_archive_md5":MD5,"mirror_archive_sha256":receipt["sha256"],
              "source":URL,"extraction":"validation images, masks and official list only",
              "protocols":["VOC20: background ignored and foreground shifted to 0..19",
                           "VOC21: 0..20 including background; ignore 255"]}
    atomic_json(base/"manifest.json",metadata)
    print(json.dumps(metadata,indent=2))
if __name__=="__main__":main()
