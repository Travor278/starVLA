"""Check StarVLA request mapping and 30-to-20 Hz action timing without a model."""

from __future__ import annotations

import argparse
import json

import numpy as np

from examples.simBenchmarks.EvoDexBench.eval_files.bridge import (
    CAMERAS,
    ROLES,
    StarVLAPolicySession,
    aligned_action_indices,
)


class _FakeClient:
    def __init__(self, embodiment: str) -> None:
        self.roles = ROLES[embodiment]
        self.requests: list[dict] = []
        self.metadata = {
            "training_data_mix": f"evodex_{embodiment}_h50_q99",
            "action_chunk_size": 50,
            "training_obs_image_size": [224, 224],
            "available_unnorm_keys": ["new_embodiment"],
            "state_keys": [
                key for role in self.roles
                for key in (f"state.{role}_ee_pose", f"state.{role}_hand_qpos")
            ],
            "action_keys": [
                key for role in self.roles
                for key in (f"action.{role}_ee_delta", f"action.{role}_hand_qpos")
            ],
        }

    def get_server_metadata(self) -> dict:
        return self.metadata

    def predict_action(self, request: dict) -> dict:
        self.requests.append(request)
        chunk = np.zeros((1, 50, 30 * len(self.roles)), dtype=np.float32)
        for role_index in range(len(self.roles)):
            offset = role_index * 30
            chunk[0, :, offset] = 0.001 * np.arange(50)
            chunk[0, :, offset + 6:offset + 30] = 0.2
        return {"data": {"actions": chunk}}

    def close(self) -> None:
        pass


def inspect(embodiment: str) -> dict[str, object]:
    roles = ROLES[embodiment]
    cameras = CAMERAS[embodiment]
    client = _FakeClient(embodiment)
    session = StarVLAPolicySession(
        embodiment=embodiment, execute_steps=2, client=client
    )
    arm_spec = {
        "translation_lower": np.full(3, -0.1),
        "translation_upper": np.full(3, 0.1),
        "rotation_constraint": {"scale_rad": np.full(3, 0.1)},
    }
    hand_spec = {"lower": np.full(24, -1.0), "upper": np.full(24, 1.0)}
    session.bind_environment({
        "manipulator_roles": roles,
        "camera_roles": cameras,
        "control_frequency_hz": 20,
        "benchmark_action_spec": {
            "action_format": "benchmark_physical_semantic",
            "manipulators": {
                role: {
                    "end_effector_delta_pose": arm_spec,
                    "hand_joint_position": hand_spec,
                }
                for role in roles
            },
        },
    })
    model_input = {
        "instruction": "Move the object safely.",
        "observation": {
            "cameras": {
                camera: {"rgb": np.full((224, 224, 3), index, dtype=np.uint8)}
                for index, camera in enumerate(cameras, start=1)
            },
            "manipulators": {
                role: {
                    "end_effector_pose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
                    "joint_position": np.zeros(31, dtype=np.float32),
                }
                for role in roles
            },
        },
    }
    try:
        first = session.act(model_input)
        if session.capture_visual_observation_after_action():
            raise AssertionError("Chunk buffer must cover the second control tick")
        second = session.act(model_input)
        if not session.capture_visual_observation_after_action():
            raise AssertionError("Chunk buffer must replan after two control ticks")
        if len(client.requests) != 1:
            raise AssertionError("Two executed actions should use one model request")
        request = client.requests[0]
        if request["normalize_state"] is not True:
            raise AssertionError("Physical state must use the server training transform")
        image_values = [int(image[0, 0, 0]) for image in request["examples"][0]["image"]]
        if image_values != [1, 2, 3]:
            raise AssertionError("Request camera order changed")
        for role in roles:
            if not np.isclose(first[role]["end_effector_delta_pose"][0], 0.0):
                raise AssertionError("First action must use checkpoint frame 0")
            if not np.isclose(second[role]["end_effector_delta_pose"][0], 0.003):
                raise AssertionError("Second action must use checkpoint frame 2 and 1.5x delta")
            if not np.allclose(second[role]["hand_joint_position"], 0.2):
                raise AssertionError("Absolute hand targets must not be scaled")
        return {
            "embodiment": embodiment,
            "camera_order": list(cameras),
            "aligned_indices_first_five": aligned_action_indices()[:5].tolist(),
            "ee_delta_scale": 1.5,
            "physical_action_dim": 30 * len(roles),
            "request_count": len(client.requests),
            "passed": True,
        }
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiment", choices=("single", "dual"), required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.embodiment), indent=2))


if __name__ == "__main__":
    main()
