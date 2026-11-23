"""Restore the checked binary assets from this repository's release parts."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]


def restore(entry):
    target = ROOT / entry["target"]
    expected = entry["sha256"]
    if target.exists():
        if sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Existing asset differs: {target}")
        print(f"Verified {entry['target']}")
        return

    digest = sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=".restore-", delete=False) as output:
            temporary = Path(output.name)
            for part in entry["parts"]:
                block = (ROOT / part).read_bytes()
                digest.update(block)
                output.write(block)
        if temporary.stat().st_size != entry["bytes"] or digest.hexdigest() != expected:
            raise RuntimeError(f"Asset parts did not match expected checksum: {target}")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Restored {entry['target']}")


def main():
    manifest = json.loads((ROOT / "release_assets/manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["assets"]:
        restore(entry)


if __name__ == "__main__":
    main()
