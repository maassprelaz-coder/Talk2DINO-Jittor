import requests
from pathlib import Path
r=requests.get("https://raw.githubusercontent.com/xb534/SED/main/datasets/pascalcontext_val.txt",timeout=30);r.raise_for_status()
p=Path(__file__).resolve().parents[1]/"01_证据/context_sources/reference_val.txt"
p.write_text(r.text)
print("Reference validation IDs:",len(r.text.splitlines()))
