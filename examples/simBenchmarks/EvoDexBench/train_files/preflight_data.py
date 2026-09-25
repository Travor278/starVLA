"""Read one real StarVLA sample from each prepared Evo-DexBench view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def inspect(data_root: Path, embodiment: str) -> dict[str, object]:
    example_root = Path(__file__).resolve().parents[1]
    config_path = (
        example_root / "train_files" / f"starvla_evodex_{embodiment}_qwenoft_h50_q99.yaml"
    )
    config = OmegaConf.load(config_path)
    config.datasets.vla_data.data_root_dir = str(data_root.resolve(strict=True))
    dataset = get_vla_dataset(data_cfg=config.datasets.vla_data)
    if len(dataset.datasets) != 1:
        raise ValueError("Preflight expects exactly one physical source dataset")
    sample = dataset.datasets[0][0]
    action = np.asarray(sample["action"])
    state = np.asarray(sample["state"])
    images = [np.asarray(image) for image in sample["image"]]
    expected_roles = 1 if embodiment == "single" else 2
    if action.shape != (50, 30 * expected_roles):
        raise ValueError(f"Unexpected normalized action shape {action.shape}")
    if state.shape != (1, 31 * expected_roles):
        raise ValueError(f"Unexpected normalized state shape {state.shape}")
    if len(images) != 3 or any(image.shape != (224, 224, 3) for image in images):
        raise ValueError("Unexpected decoded RGB camera count or shape")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Non-finite normalized state or action")
    view_path = data_root / f"evodex_{embodiment}"
    view = json.loads((view_path / "meta/starvla_view.json").read_text())
    return {
        "embodiment": embodiment,
        "dataset": str(view_path.resolve()),
        "sample_index": 0,
        "episode_count": int(dataset.datasets[0].lerobot_info_meta["total_episodes"]),
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "image_shapes": [list(image.shape) for image in images],
        "camera_order": view["camera_order"],
        "action_min": float(action.min()),
        "action_max": float(action.max()),
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--embodiment", choices=("single", "dual"), required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.data_root, args.embodiment), indent=2))


if __name__ == "__main__":
    main()
