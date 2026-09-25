"""LeRobot v3 registrations for Evo-DexBench physical semantic actions."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class _EvoDexConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    observation_indices = [0]
    action_indices = list(range(50))
    roles: tuple[str, ...] = ()
    video_keys: list[str] = []

    @property
    def state_keys(self) -> list[str]:
        return [key for role in self.roles for key in
                (f"state.{role}_ee_pose", f"state.{role}_hand_qpos")]

    @property
    def action_keys(self) -> list[str]:
        return [key for role in self.roles for key in
                (f"action.{role}_ee_delta", f"action.{role}_hand_qpos")]

    @property
    def state_key_dims(self) -> dict[str, int]:
        return {key: (7 if key.endswith("ee_pose") else 24)
                for key in self.state_keys}

    @property
    def action_key_dims(self) -> dict[str, int]:
        return {key: (6 if key.endswith("ee_delta") else 24)
                for key in self.action_keys}

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=["annotation.human.action.task_description"],
            ),
        }

    def transform(self) -> ComposedModalityTransform:
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
        ])


class EvoDexSingleConfig(_EvoDexConfig):
    roles = ("right",)
    video_keys = [
        "video.context", "video.interaction", "video.wrist_right",
    ]


class EvoDexDualConfig(_EvoDexConfig):
    roles = ("left", "right")
    video_keys = [
        "video.context", "video.wrist_left", "video.wrist_right",
    ]


ROBOT_TYPE_CONFIG_MAP = {
    "evodex_single_h50_q99": EvoDexSingleConfig(),
    "evodex_dual_h50_q99": EvoDexDualConfig(),
}

DATASET_NAMED_MIXTURES = {
    "evodex_single_h50_q99": [
        ("evodex_single", 1.0, "evodex_single_h50_q99"),
    ],
    "evodex_dual_h50_q99": [
        ("evodex_dual", 1.0, "evodex_dual_h50_q99"),
    ],
}
