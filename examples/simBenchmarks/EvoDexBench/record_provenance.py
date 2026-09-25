"""Record immutable code, data-view, and checkpoint identities for a run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _files(path: Path) -> dict[str, str]:
    if path.is_file():
        return {path.name: _sha256(path)}
    return {
        item.relative_to(path).as_posix(): _sha256(item)
        for item in sorted(path.rglob("*")) if item.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evodex-root", type=Path, required=True)
    parser.add_argument("--data-view", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    starvla_root = Path(__file__).resolve().parents[3]
    evodex_root = args.evodex_root.expanduser().resolve(strict=True)
    data_view = args.data_view.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True) if args.checkpoint else None
    if args.output.exists():
        raise FileExistsError(args.output)
    view_manifest = data_view / "meta/starvla_view.json"
    result = {
        "run_id": args.run_id,
        "starvla_commit": _git(starvla_root, "rev-parse", "HEAD"),
        "starvla_dirty": bool(_git(starvla_root, "status", "--porcelain")),
        "evodex_commit": _git(evodex_root, "rev-parse", "HEAD"),
        "evodex_dirty": bool(_git(evodex_root, "status", "--porcelain")),
        "data_view": str(data_view),
        "data_view_manifest_sha256": _sha256(view_manifest),
        "data_source": json.loads(view_manifest.read_text(encoding="utf-8")),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_file_sha256": _files(checkpoint) if checkpoint else {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
