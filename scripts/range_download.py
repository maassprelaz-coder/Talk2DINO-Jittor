"""Resumable bounded-memory HTTP range downloads with ETag and length guards."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def session():
    s=requests.Session()
    retry=Retry(total=4,backoff_factor=1,status_forcelist=[429,500,502,503,504],
                allowed_methods=["GET","HEAD"])
    s.mount("https://",HTTPAdapter(max_retries=retry))
    return s

def probe(url):
    with session() as client:
        response=client.get(url,headers={"Range":"bytes=0-0","Accept-Encoding":"identity"},
                            timeout=(20,60),stream=True)
        response.raise_for_status()
        if response.status_code!=206 or "Content-Range" not in response.headers:
            response.close()
            raise RuntimeError("Server must support verified HTTP byte ranges: "+url)
        size=int(response.headers["Content-Range"].split("/")[-1])
        etag=response.headers.get("ETag")
        response.close()
    return {"url":url,"size":size,"etag":etag}

def atomic_json(path,data):
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n")
    temporary.replace(path)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def download(url,destination,chunk_size=32*1024*1024,workers=4,max_chunks=None):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    current=probe(url)
    receipt=destination.with_name(destination.name+".download.json")
    partial=destination.with_name(destination.name+".part")
    count=math.ceil(current["size"]/chunk_size)
    if receipt.exists():
        state=json.loads(receipt.read_text())
        if any(state.get(k)!=current[k] for k in ["url","size","etag"]):
            raise RuntimeError("Remote object changed; review before reusing partial download")
        if state["chunk_size"]!=chunk_size:
            raise RuntimeError("Chunk size changed; keep original resume settings")
        if state.get("complete") and destination.exists() and destination.stat().st_size==current["size"]:
            return state
    else:
        if destination.exists():
            raise RuntimeError("Destination exists without a verified download receipt")
        state={**current,"chunk_size":chunk_size,"done":[],"complete":False}
        atomic_json(receipt,state)
    if not partial.exists() and destination.exists() and len(state["done"])==count and destination.stat().st_size==current["size"]:
        state.update(complete=True,sha256=sha256(destination))
        atomic_json(receipt,state)
        return state
    if not partial.exists():
        if state["done"]:
            raise RuntimeError("Partial file missing despite completed chunks")
        with partial.open("wb") as f:f.truncate(current["size"])
    if partial.stat().st_size!=current["size"]:
        raise RuntimeError("Partial download size mismatch")
    done=set(state["done"])
    pending=[index for index in range(count) if index not in done]
    if max_chunks is not None:pending=pending[:max_chunks]
    lock=threading.Lock()
    descriptor=os.open(partial,os.O_RDWR)
    start_time=time.monotonic(); last_print=[0.]
    local=threading.local()
    def transfer(index):
        if not hasattr(local,"client"):local.client=session()
        start=index*chunk_size
        end=min(start+chunk_size,current["size"])-1
        headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"}
        if current["etag"]:headers["If-Match"]=current["etag"]
        response=local.client.get(url,headers=headers,timeout=(20,120))
        response.raise_for_status()
        expected=f"bytes {start}-{end}/{current['size']}"
        if response.status_code!=206 or response.headers.get("Content-Range")!=expected:
            raise RuntimeError("Unexpected range response")
        payload=response.content
        if len(payload)!=end-start+1:raise RuntimeError("Truncated range response")
        offset=0
        while offset<len(payload):
            written=os.pwrite(descriptor,payload[offset:],start+offset)
            if written<=0:raise RuntimeError("Short write")
            offset+=written
        with lock:
            done.add(index)
            state["done"]=sorted(done)
            atomic_json(receipt,state)
            elapsed=time.monotonic()-start_time
            if elapsed-last_print[0]>=10 or len(done)==count:
                completed=min(len(done)*chunk_size,current["size"])
                print(f"{destination.name}: {len(done)}/{count} chunks; {completed/1e9:.2f}/{current['size']/1e9:.2f} GB",flush=True)
                last_print[0]=elapsed
        return index
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(transfer,index) for index in pending]):
                future.result()
        os.fsync(descriptor)
    finally:os.close(descriptor)
    if len(done)==count:
        partial.replace(destination)
        state.update(complete=True,sha256=sha256(destination))
        atomic_json(receipt,state)
    return state
