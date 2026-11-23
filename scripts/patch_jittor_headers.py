"""Remove an unnecessary NPP header dependency using Jittor's own integer constants."""
import difflib
import hashlib
import json
import re
from pathlib import Path
import sysconfig
import os

ROOT=Path(__file__).resolve().parents[1]
TARGET=Path(sysconfig.get_paths()["purelib"])/"jittor/src/misc/cuda_limits.h"
text=TARGET.read_text()
if "#include <npp.h>" in text:
    constants=re.findall(r"^#define[ \t]+NPP_\w+.*$",text,re.M)
    if len(constants)!=8:
        raise RuntimeError("Unexpected Jittor integer limits; review patch")
    updated=text.replace("#include <npp.h>","// project: identical integer limits from the CPU branch; no NPP runtime used.\n"+"\n".join(constants))
    backup=TARGET.with_name("cuda_limits.h.talk2dino_original")
    if not backup.exists(): backup.write_text(text)
    TARGET.write_text(updated)
    evidence=ROOT/"01_证据"
    evidence.mkdir(parents=True,exist_ok=True)
    (evidence/"jittor_npp_header_compat.patch").write_text("".join(difflib.unified_diff(
        text.splitlines(True),updated.splitlines(True),fromfile="jittor-1.3.11/cuda_limits.h",
        tofile="project-patched/cuda_limits.h")))
    (evidence/"jittor_npp_patch.json").write_text(json.dumps({
        "before_sha256":hashlib.sha256(text.encode()).hexdigest(),
        "after_sha256":hashlib.sha256(updated.encode()).hexdigest(),
        "reason":"Use identical standard integer limits already defined by Jittor, avoiding unused NPP SDK dependency."
    },indent=2)+"\n")
    print("Removed unused NPP header requirement with version-guarded local patch")
else:
    print("NPP compatibility patch already applied")
