"""Reset one Evo-DexBench task and execute one bounded neutral action."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from dex_benchmark.integration.model_input import build_model_input, instruction_from_env
from dex_benchmark.policies.observation import as_finite_numpy, required_value
from dex_benchmark.tasks import make_task_env

ROLES = {"single": ("right",), "dual": ("left", "right")}
CAMERAS = {
    "single": ("context", "interaction", "wrist_right"),
    "dual": ("context", "wrist_left", "wrist_right"),
}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return bool(np.isfinite(as_finite_numpy(value, name="native_action")).all())


def inspect(task_id: str, embodiment: str, seed: int, *, cpu_render: bool) -> dict[str, object]:
    env_kwargs: dict[str, Any] = {
        "obs_mode": "rgb",
        "control_mode": "arm_pd_ee_delta_pose_hand_pd_joint_pos",
        "render_mode": None,
    }
    if cpu_render:
        env_kwargs.update(sim_backend="cpu", render_backend="cpu")
    env = make_task_env(task_id, **env_kwargs)
    try:
        observation, _ = env.reset(seed=seed)
        get_metadata = env.get_wrapper_attr("get_episode_metadata")
        metadata = get_metadata()
        roles = ROLES[embodiment]
        cameras = CAMERAS[embodiment]
        if tuple(metadata["manipulator_roles"]) != roles:
            raise ValueError("Task manipulator roles differ from the requested embodiment")
        if metadata["control_frequency_hz"] != 30:
            raise ValueError(
                "Task control frequency differs from dataset 30 Hz: "
                f"got {metadata['control_frequency_hz']} Hz"
            )
        if not set(cameras).issubset(metadata["camera_roles"]):
            raise ValueError("Task observation lacks a required camera")
        model_input = build_model_input(observation, instruction_from_env(env))
        images = required_value(model_input["observation"], "cameras")
        for camera in cameras:
            image = as_finite_numpy(required_value(images, camera, "rgb"), name=camera)
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{camera} has unexpected image shape or dtype")
        manipulators = required_value(model_input["observation"], "manipulators")
        action = {}
        for role in roles:
            qpos = as_finite_numpy(
                required_value(manipulators, role, "joint_position"),
                name=f"{role}.joint_position",
            ).astype(np.float32)
            if qpos.shape != (31,):
                raise ValueError(f"{role} joint state must contain 7 arm + 24 hand values")
            action[role] = {
                "end_effector_delta_pose": np.zeros(6, dtype=np.float32),
                "hand_joint_position": qpos[7:].copy(),
            }
        encode_action = env.get_wrapper_attr("encode_action")
        native = encode_action(action, bounds_policy="error")
        if not _finite_tree(native):
            raise ValueError("Native controller action contains non-finite values")
        _, reward, terminated, truncated, _ = env.step(native)
        return {
            "task_id": task_id,
            "embodiment": embodiment,
            "seed": seed,
            "camera_order": list(cameras),
            "control_hz": 30,
            "cpu_render": cpu_render,
            "one_step_reward_finite": _finite_tree(reward),
            "terminated": bool(np.asarray(terminated).any()),
            "truncated": bool(np.asarray(truncated).any()),
            "one_neutral_step_passed": True,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--embodiment", choices=("single", "dual"), required=True)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--cpu-render", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            inspect(args.task_id, args.embodiment, args.seed, cpu_render=args.cpu_render),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
