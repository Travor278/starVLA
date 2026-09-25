"""Check saved StarVLA normalization against a real Evo-DexBench row."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from examples.simBenchmarks.EvoDexBench.train_files.prepare_dataset_view import modality_for
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def inspect(data_root: Path, embodiment: str) -> dict[str, object]:
    benchmark_root = Path(__file__).resolve().parents[1]
    config_path = (
        benchmark_root / "train_files" / f"starvla_evodex_{embodiment}_qwenoft_h50_q99.yaml"
    )
    config = OmegaConf.load(config_path)
    config.datasets.vla_data.data_root_dir = str(data_root.resolve(strict=True))
    mixture = get_vla_dataset(data_cfg=config.datasets.vla_data)
    dataset = mixture.datasets[0]
    trajectory = dataset.get_trajectory_data(int(dataset.trajectory_ids[0]))
    row = trajectory.iloc[0]
    raw_state = np.asarray(row["observation.state"], dtype=np.float32)
    raw_action = np.asarray(row["action"], dtype=np.float32)
    modality = modality_for(embodiment)
    state = {
        f"state.{name}": raw_state[part["start"]:part["end"]][None, :]
        for name, part in modality["state"].items()
    }
    state_flat = np.concatenate(list(state.values()), axis=-1)
    action_dim = 30 if embodiment == "single" else 60
    if raw_action.shape != (action_dim,):
        raise ValueError(f"Raw physical action shape {raw_action.shape}")

    with tempfile.TemporaryDirectory(prefix="evodex_starvla_processor_") as directory:
        run_dir = Path(directory)
        shutil.copy2(config_path, run_dir / "config.yaml")
        mixture.save_dataset_statistics(run_dir / "dataset_statistics.json")
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        fake_checkpoint = checkpoint_dir / "processor_only.pt"
        fake_checkpoint.touch()
        processor = PolicyNormProcessor(str(fake_checkpoint))
        normalized = processor.apply_state(state_flat)
        expected_parts = dataset.transforms.apply(dict(state))
        expected = np.concatenate(
            [np.asarray(expected_parts[key]) for key in processor.state_keys], axis=-1
        )
        state_error = float(np.max(np.abs(normalized - expected)))
        if normalized.shape != expected.shape or not np.isfinite(normalized).all():
            raise ValueError("State processor output shape or finite check failed")
        if state_error > 1e-5:
            raise ValueError(f"State processor differs from training transform by {state_error}")

        stats = json.loads((run_dir / "dataset_statistics.json").read_text())
        action_stats = stats[processor.unnorm_key]["action"]
        midpoint = 0.5 * (
            np.asarray(action_stats["q01"], dtype=np.float32)
            + np.asarray(action_stats["q99"], dtype=np.float32)
        )
        physical = processor.unapply_actions(np.zeros((1, action_dim), dtype=np.float32))
        action_error = float(np.max(np.abs(physical[0] - midpoint)))
        if physical.shape != (1, action_dim) or not np.isfinite(physical).all():
            raise ValueError("Action processor output shape or finite check failed")
        if action_error > 1e-5:
            raise ValueError(f"Action inverse differs from q99 midpoint by {action_error}")
        training_action = dataset.transforms.unapply({
            f"action.{name}": torch.zeros((1, part["end"] - part["start"]))
            for name, part in modality["action"].items()
        })
        expected_action = np.concatenate(
            [np.asarray(training_action[key]) for key in processor.action_keys], axis=-1
        )
        training_action_error = float(np.max(np.abs(physical - expected_action)))
        if training_action_error > 1e-5:
            raise ValueError(
                "Action processor differs from the training transform by "
                f"{training_action_error}"
            )

    return {
        "embodiment": embodiment,
        "state_dim": int(state_flat.shape[-1]),
        "action_dim": action_dim,
        "physical_action_abs_max": float(np.abs(raw_action).max()),
        "state_training_processor_max_abs_error": state_error,
        "action_inverse_midpoint_max_abs_error": action_error,
        "action_training_processor_max_abs_error": training_action_error,
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
