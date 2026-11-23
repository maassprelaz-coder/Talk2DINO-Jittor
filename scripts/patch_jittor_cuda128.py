"""Version-guarded local fixes for Jittor 1.3.11 + CUDA 12.8 / Blackwell."""
import difflib
import hashlib
import json
from pathlib import Path
import sysconfig
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
TARGET=Path(sysconfig.get_paths()["purelib"])/"jittor/compiler.py"
BEFORE="cache_path = os.path.join(cache_path, cu)\n    sys.path.append(cache_path)"
AFTER="cache_path = os.path.join(cache_path, cu)\n    sys.path.insert(0, cache_path)  # project: prefer CUDA core over previously cached CPU core"
OLD="    max_arch = 90\n    if nvcc_version < [11,]:"
NEW="    max_arch = 120 if nvcc_version >= [12,8] else 90\n    if nvcc_version < [11,]:"

def main():
    text=TARGET.read_text()
    evidence=ROOT/"01_证据"
    evidence.mkdir(parents=True,exist_ok=True)
    if AFTER in text and NEW in text:
        print("Version-guarded CUDA compatibility fixes already applied")
        return
    if text.count(BEFORE)!=1 or text.count(OLD)!=1:
        raise RuntimeError("Unexpected Jittor source; patch must be reviewed")
    original=text
    updated=text.replace(BEFORE,AFTER).replace(OLD,NEW)
    backup=TARGET.with_name("compiler.py.talk2dino_original")
    if not backup.exists(): backup.write_text(original)
    TARGET.write_text(updated)
    diff="".join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),
                                    fromfile="jittor-1.3.11/compiler.py",tofile="project-patched/compiler.py"))
    (evidence/"jittor_1.3.11_cuda128_compat.patch").write_text(diff)
    report={"package":"jittor==1.3.11.0","target":str(TARGET),
            "before_sha256":hashlib.sha256(original.encode()).hexdigest(),
            "after_sha256":hashlib.sha256(updated.encode()).hexdigest(),
            "fixes":["CUDA module search precedence after a CPU import/cache",
                     "allow sm_120 when nvcc is at least 12.8"],
            "scope":"project-private venv; no system package edited"}
    (evidence/"jittor_compat_patch.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
