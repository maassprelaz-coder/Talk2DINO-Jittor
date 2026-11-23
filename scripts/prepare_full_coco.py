"""Download and verify the complete official COCO 2014 image splits."""
import argparse
import json
from pathlib import Path
import shutil
import time
import zipfile
import zlib
from range_download import download,atomic_json,sha256

BASE="https://s3.amazonaws.com/images.cocodataset.org"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args();root=args.root.resolve()
    data=root/"data/coco2014"
    archives=root/"data/archives"
    archives.mkdir(parents=True,exist_ok=True)
    state_path=data/"full_preparation.json"
    progress=json.loads(state_path.read_text()) if state_path.exists() else {"complete_splits":[],"splits":{}}
    for split in ["train","val"]:
        annotation=data/f"annotations/captions_{split}2014.json"
        document=json.loads(annotation.read_text())
        expected={item["file_name"] for item in document["images"]}
        archive=archives/f"{split}2014.zip"
        if split in progress["complete_splits"]:
            print(f"{split} previously prepared and verified",flush=True)
            continue
        reserve=30*1024**3
        if shutil.disk_usage(archives).free<reserve+45*1024**3:
            raise RuntimeError("Insufficient free space for archives, extracted data and reserved headroom")
        record=download(f"{BASE}/zips/{split}2014.zip",archive,workers=args.workers)
        if not record["complete"]:raise RuntimeError("Download incomplete")
        destination=data/f"{split}2014"
        destination.mkdir(exist_ok=True)
        started=time.monotonic(); last_print=started
        verified=0
        with zipfile.ZipFile(archive) as source:
            infos={Path(item.filename).name:item for item in source.infolist()
                   if item.filename.startswith(f"{split}2014/") and item.filename.lower().endswith(".jpg")}
            if set(infos)!=expected:
                raise RuntimeError("Image archive and official caption image sets differ")
            for name in sorted(expected):
                item=infos[name]
                target=destination/name
                already=False
                if target.exists() and target.stat().st_size==item.file_size:
                    already=(zlib.crc32(target.read_bytes()) & 0xffffffff)==item.CRC
                if not already:
                    payload=source.read(item)  # ZipFile validates CRC before returning.
                    temporary=target.with_suffix(".jpg.tmp")
                    temporary.write_bytes(payload);temporary.replace(target)
                verified+=1
                if time.monotonic()-last_print>=15:
                    print(f"{split} extraction: {verified}/{len(expected)} CRC-verified images",flush=True)
                    last_print=time.monotonic()
        progress["splits"][split]={"images":len(expected),"captions":len(document["annotations"]),
                                  "annotation_sha256":sha256(annotation),"archive":record}
        progress["complete_splits"].append(split)
        atomic_json(state_path,progress)
        print(f"{split} ready: {len(expected)} images, {len(document['annotations'])} captions",flush=True)
    progress["complete"]=True
    atomic_json(state_path,progress)
    print("Full official COCO 2014 preparation complete",flush=True)

if __name__=="__main__":main()
