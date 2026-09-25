"""Create a metadata-only StarVLA view of an Evo-DexBench LeRobot v3 export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CAMERAS = {
    "single": ("context", "interaction", "wrist_right"),
    "dual": ("context", "wrist_left", "wrist_right"),
}
ROLES = {"single": ("right",), "dual": ("left", "right")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def modality_for(embodiment: str) -> dict[str, Any]:
    roles = ROLES[embodiment]
    state: dict[str, Any] = {}
    action: dict[str, Any] = {}
    for index, role in enumerate(roles):
        state_offset = 111 * index
        action_offset = 30 * index
        state[f"{role}_ee_pose"] = {
            "start": state_offset, "end": state_offset + 7,
            "original_key": "observation.state",
        }
        state[f"{role}_hand_qpos"] = {
            "start": state_offset + 49, "end": state_offset + 73,
            "original_key": "observation.state",
        }
        action[f"{role}_ee_delta"] = {
            "start": action_offset, "end": action_offset + 6,
            "original_key": "action",
            "absolute": False,
        }
        action[f"{role}_hand_qpos"] = {
            "start": action_offset + 6, "end": action_offset + 30,
            "original_key": "action",
        }
    return {
        "state": state,
        "action": action,
        "video": {
            camera: {"original_key": f"observation.images.{camera}"}
            for camera in CAMERAS[embodiment]
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def prepare(source: Path, data_root: Path, embodiment: str) -> Path:
    source = source.expanduser().resolve(strict=True)
    data_root = data_root.expanduser().resolve()
    name = f"evodex_{embodiment}"
    target = data_root / name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite dataset view: {target}")
    info_path = source / "meta/info.json"
    lineage_path = source / "meta/dex_benchmark.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    roles = ROLES[embodiment]
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 30:
        raise ValueError("Evo-DexBench source must be LeRobot v3.0 at 30 Hz")
    features = info["features"]
    if features["observation.state"]["shape"] != [111 * len(roles)]:
        raise ValueError("Unexpected canonical state dimension")
    if features["action"]["shape"] != [30 * len(roles)]:
        raise ValueError("Unexpected physical action dimension")
    if not all(f"observation.images.{key}" in features for key in CAMERAS[embodiment]):
        raise ValueError("Missing required camera stream")
    episodes = lineage.get("episodes", [])
    if len(episodes) != info.get("total_episodes") or not episodes:
        raise ValueError("Episode lineage count differs from LeRobot metadata")
    if any(tuple(row.get("manipulator_roles", ())) != roles for row in episodes):
        raise ValueError("Episode embodiment differs from requested view")
    if any(row.get("control_hz") != 30 for row in episodes):
        raise ValueError("Episode control frequency differs from 30 Hz")
    if any(row.get("action_feature_order") != [
        feature for role in roles for feature in
        (f"{role}.end_effector_delta_pose", f"{role}.hand_joint_position")
    ] for row in episodes):
        raise ValueError("Episode physical action order differs from the adapter")
    if any(row.get("generalization", {}).get("range_version") == "evaluation_v1"
           for row in episodes):
        raise ValueError("Evaluation-profile episodes cannot enter a training view")
    for relative in ("data", "videos", "meta/stats.json", "meta/tasks.parquet", "meta/episodes"):
        if not (source / relative).exists():
            raise FileNotFoundError(source / relative)

    (target / "meta").mkdir(parents=True)
    for name in ("data", "videos"):
        (target / name).symlink_to(source / name, target_is_directory=True)
    for name in ("info.json", "stats.json", "tasks.parquet", "episodes", "dex_benchmark.json"):
        origin = source / "meta" / name
        (target / "meta" / name).symlink_to(origin, target_is_directory=origin.is_dir())
    (target / "meta/modality.json").write_text(
        json.dumps(modality_for(embodiment), indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "source": str(source),
        "embodiment": embodiment,
        "source_info_sha256": _sha256(info_path),
        "source_lineage_sha256": _sha256(lineage_path),
        "source_stats_sha256": _sha256(source / "meta/stats.json"),
        "episode_ids": [str(row["episode_id"]) for row in episodes],
        "camera_order": list(CAMERAS[embodiment]),
        "state_dim": 31 * len(roles),
        "action_dim": 30 * len(roles),
        "control_hz": 30,
    }
    (target / "meta/starvla_view.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embodiment", choices=tuple(ROLES), required=True)
    args = parser.parse_args()
    print(prepare(args.source, args.data_root, args.embodiment))


if __name__ == "__main__":
    main()
