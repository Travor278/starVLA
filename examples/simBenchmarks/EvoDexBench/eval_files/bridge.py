"""Thin StarVLA WebSocket policy session for Evo-DexBench's own evaluator."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
from dex_benchmark.integration.model_input import ModelInput
from dex_benchmark.integration.policy_features import decode_semantic_action
from dex_benchmark.policies.observation import compact_runtime_state, required_value

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

ROLES = {"single": ("right",), "dual": ("left", "right")}
CAMERAS = {
    "single": ("context", "interaction", "wrist_right"),
    "dual": ("context", "wrist_left", "wrist_right"),
}


class StarVLAPolicySession:
    """Return physical semantic actions; Evo-DexBench encodes and scores them."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 10093,
        embodiment: str,
        execute_steps: int = 10,
        unnorm_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if embodiment not in ROLES:
            raise ValueError("embodiment must be 'single' or 'dual'")
        if not 1 <= execute_steps <= 50:
            raise ValueError("execute_steps must be between 1 and 50")
        self.embodiment = embodiment
        self.roles = ROLES[embodiment]
        self.camera_roles = CAMERAS[embodiment]
        self.action_dim = 30 * len(self.roles)
        self.state_dim = 31 * len(self.roles)
        self.execute_steps = execute_steps
        self.unnorm_key = unnorm_key
        self.client = client or WebsocketClientPolicy(host, port)
        self._actions: deque[np.ndarray] = deque()
        self._action_spec: Mapping[str, Any] | None = None
        self._check_server_metadata(self.client.get_server_metadata())

    def _check_server_metadata(self, metadata: Mapping[str, Any]) -> None:
        expected_mix = f"evodex_{self.embodiment}_h50_q99"
        if metadata.get("training_data_mix") != expected_mix:
            raise ValueError(f"Server data mix must be {expected_mix!r}")
        if metadata.get("action_chunk_size") != 50:
            raise ValueError("Server action chunk must contain 50 control ticks")
        if metadata.get("training_obs_image_size") != [224, 224]:
            raise ValueError("Server training images must be 224 x 224")
        available = metadata.get("available_unnorm_keys", ())
        if self.unnorm_key is None:
            if len(available) != 1:
                raise ValueError("Specify unnorm_key for a multi-dataset checkpoint")
            self.unnorm_key = str(available[0])
        elif self.unnorm_key not in available:
            raise ValueError(f"Unnormalization key {self.unnorm_key!r} not in {available!r}")
        expected_state = [
            feature for role in self.roles for feature in
            (f"state.{role}_ee_pose", f"state.{role}_hand_qpos")
        ]
        expected_action = [
            feature for role in self.roles for feature in
            (f"action.{role}_ee_delta", f"action.{role}_hand_qpos")
        ]
        if metadata.get("state_keys") != expected_state:
            raise ValueError("Server state key order differs from Evo-DexBench")
        if metadata.get("action_keys") != expected_action:
            raise ValueError("Server action key order differs from Evo-DexBench")

    def bind_environment(self, metadata: Mapping[str, Any]) -> None:
        if tuple(metadata.get("manipulator_roles", ())) != self.roles:
            raise ValueError("Environment manipulator order differs from checkpoint")
        if not set(self.camera_roles).issubset(metadata.get("camera_roles", ())):
            raise ValueError("Environment lacks a required policy camera")
        if metadata.get("control_frequency_hz") != 30:
            raise ValueError("Environment control frequency must be 30 Hz")
        spec = metadata.get("benchmark_action_spec")
        if not isinstance(spec, Mapping) or spec.get("action_format") != "benchmark_physical_semantic":
            raise ValueError("Environment lacks the physical semantic action contract")
        if tuple(spec.get("manipulators", ())) != self.roles:
            raise ValueError("Environment physical action roles differ")
        self._action_spec = spec

    def _example(self, model_input: ModelInput) -> dict[str, Any]:
        observation = model_input["observation"]
        cameras = required_value(observation, "cameras")
        manipulators = required_value(observation, "manipulators")
        images: list[np.ndarray] = []
        for camera in self.camera_roles:
            image = np.asarray(required_value(cameras, camera, "rgb"))
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{camera} must be uint8 RGB [224,224,3]")
            images.append(np.ascontiguousarray(image))
        state = compact_runtime_state(manipulators, self.roles)
        if state.shape != (self.state_dim,):
            raise ValueError(f"Runtime state has shape {state.shape}")
        instruction = str(model_input["instruction"]).strip()
        if not instruction:
            raise ValueError("Empty task instruction")
        return {"image": images, "state": state[None, :], "lang": instruction}

    def _check_physical_action(self, action: np.ndarray) -> None:
        if self._action_spec is None:
            raise RuntimeError("Bind environment metadata before inference")
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Predicted physical action has wrong shape or non-finite values")
        for index, role in enumerate(self.roles):
            block = action[index * 30:(index + 1) * 30]
            role_spec = self._action_spec["manipulators"][role]
            arm = role_spec["end_effector_delta_pose"]
            hand = role_spec["hand_joint_position"]
            translation_bound = np.maximum(
                np.abs(arm["translation_lower"]), np.abs(arm["translation_upper"])
            )
            rotation_bound = np.asarray(arm["rotation_constraint"].get("scale_rad", [1, 1, 1]))
            # The frozen evaluator clips to controller bounds. Reject grossly
            # invalid model outputs before they reach the simulator.
            if np.any(np.abs(block[:3]) > np.maximum(1.0, 10 * translation_bound)):
                raise ValueError(f"{role} EE translation exceeds the safety guard")
            if np.any(np.abs(block[3:6]) > np.maximum(1.0, 10 * rotation_bound)):
                raise ValueError(f"{role} EE rotation exceeds the safety guard")
            if np.any(block[6:] < np.asarray(hand["lower"]) - 1.0) or np.any(
                block[6:] > np.asarray(hand["upper"]) + 1.0
            ):
                raise ValueError(f"{role} hand target exceeds the safety guard")

    def reset(self) -> None:
        self._actions.clear()

    def act(self, model_input: ModelInput) -> Mapping[str, Mapping[str, np.ndarray]]:
        if not self._actions:
            response = self.client.predict_action({
                "examples": [self._example(model_input)],
                "unnorm_key": self.unnorm_key,
                "normalize_state": True,
                "do_sample": False,
            })
            chunk = np.asarray(response["data"]["actions"], dtype=np.float32)
            if chunk.shape != (1, 50, self.action_dim) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid physical action chunk {chunk.shape}")
            for action in chunk[0, :self.execute_steps]:
                self._check_physical_action(action)
                self._actions.append(action)
        return decode_semantic_action(self._actions.popleft(), self.embodiment)

    def capture_visual_observation_after_action(self) -> bool:
        return not self._actions

    def capture_lowdim_observation_after_action(self) -> bool:
        return not self._actions

    def requires_control_period_tactile(self) -> bool:
        return False

    def close(self) -> None:
        self.client.close()
