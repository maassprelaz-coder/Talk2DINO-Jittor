"""Prepare Context59/60 val using published, checksum-verified Encoding masks."""
import argparse,ast,hashlib,json,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from range_download import download,atomic_json,sha256

URL="https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth"
EXPECTED_SHA1="ebedc94247ec616c57b9a2df15091784826a7b0c"
def mapping(path):
    candidates=[n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.List) and len(n.elts)==60]
    if len(candidates)!=1:raise RuntimeError("Unexpected source mapping structure")
    return sorted(ast.literal_eval(candidates[0]))
def class_names(path):
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="PascalContextDataset59")
    assignment=next(n for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CLASSES" for t in n.targets))
    return ast.literal_eval(assignment.value)
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    sources=root/"01_证据/context_sources"
    if mapping(sources/"mapping_source.py")!=mapping(sources/"mmseg_mapping.py"):
        raise RuntimeError("Encoding mask mapping differs from pinned MMSeg converter")
    classes=class_names(sources/"mmseg_classes.py")
    archive=root/"data/archives/context_val.pth"
    receipt=download(URL,archive,workers=4)
    digest=hashlib.sha1()
    with archive.open("rb") as stream:
        for part in iter(lambda:stream.read(16*1024*1024),b""):digest.update(part)
    if digest.hexdigest()!=EXPECTED_SHA1:raise RuntimeError("Context archive differs from published validation mask checksum")
    # Only allow Pillow's plain image container, never arbitrary pickle globals.
    with torch.serialization.safe_globals([Image.Image]):
        masks=torch.load(archive,map_location="cpu",weights_only=True)
    if not isinstance(masks,dict) or len(masks)!=5104:raise RuntimeError("Unexpected Context validation dictionary")
    base=root/"data/VOCdevkit/VOC2010";mask_dir=base/"SegmentationClassContext";image_dir=base/"JPEGImages"
    mask_dir.mkdir(parents=True,exist_ok=True);image_dir.mkdir(exist_ok=True)
    names=[];all_labels=set()
    with tarfile.open(root/"data/archives/VOCtrainval_11-May-2012.tar","r:") as images:
        for identifier,label in sorted(masks.items()):
            numeric=str(identifier)
            if len(numeric)!=10 or not numeric.isdigit():raise RuntimeError("Unexpected Context image ID")
            name=numeric[:4]+"_"+numeric[4:];names.append(name)
            array=np.asarray(label)
            if array.ndim!=2 or array.dtype!=np.uint8:raise RuntimeError("Unexpected Context mask representation")
            labels=set(np.unique(array).tolist());all_labels.update(labels)
            if not labels.issubset(set(range(60))):raise RuntimeError("Context labels outside 0..59")
            target=image_dir/(name+".jpg")
            if not target.exists():
                payload=images.extractfile("VOCdevkit/VOC2012/JPEGImages/"+name+".jpg").read()
                target.write_bytes(payload)
            with Image.open(target) as image:
                if image.size!=label.size:raise RuntimeError("Context image/mask dimensions differ")
            label.save(mask_dir/(name+".png"))
    reference=(sources/"reference_val.txt").read_text().splitlines()
    if set(reference)!=set(names) or len(reference)!=len(names):raise RuntimeError("Context IDs differ from independent published validation list")
    split=base/"ImageSets/SegmentationContext/val.txt";split.parent.mkdir(parents=True,exist_ok=True)
    split.write_text("".join(name+"\n" for name in names))
    atomic_json(base/"context_manifest.json",{"dataset":"PASCAL Context","split":"validation","images":5104,
        "source":URL,"published_archive_sha1":EXPECTED_SHA1,"archive_sha256":receipt["sha256"],
        "source_mapping_matches_mmseg_v030":True,"independent_5104_validation_ids_equal":True,"raw_labels":sorted(all_labels),"classes59":list(classes),
        "image_source":"reused byte-identical JPEG members from verified VOC2012 archive",
        "mask_preparation":"Encoding author-prepared dictionary, saved as indexed PNG; no resizing",
        "provenance":{p.name:sha256(p) for p in sources.iterdir() if p.is_file()},
        "protocols":["Context59: label 0 ignored, 1..59 shifted down","Context60: label 0 background included"]})
    print((base/"context_manifest.json").read_text(),flush=True)
if __name__=="__main__":main()
