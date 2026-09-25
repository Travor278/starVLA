# Evo-DexBench: first-person dexterous generalization and safety evaluation

This example registers Evo-DexBench LeRobot v3 training data with StarVLA and
connects StarVLA's existing WebSocket policy server to Evo-DexBench's fixed
simulator and scoring protocol. Evo-DexBench supplies the tasks, assets,
physical action encoder, episode schedule, and scores. StarVLA already has a
RoboDojo dexterous benchmark integration; this example adds Evo-DexBench's
egocentric observations and its own generalization and safety conditions.

## Pinned sources and external assets

The initial integration targets StarVLA `starVLA_dev` at
`4507931a625c844404c4a76128fb116536d8ca7c` and Evo-DexBench `main` at
`36d58021779f624006df56806b8cd9d174947a69`. The StarVLA stable branch
at inspection was `3422b9f2387b6f682cf02802904a77b23ab13afd`; it uses the
older `examples/<benchmark>` layout. Pin both repositories for a run and record
the actual commit IDs again if either moves.

Install StarVLA using its root README, and install Evo-DexBench and its
ManiSkill/SAPIEN assets using the pinned Evo-DexBench README. Keep policy
serving and simulation in separate environments if their dependencies differ.
The data, model weights, simulator assets, secrets, and run logs live outside
the repository. Training source data must be Evo-DexBench's 30 Hz LeRobot v3
export with `meta/dex_benchmark.json`; evaluation needs Evo-DexBench's task
assets. This integration does not redistribute either.

| Contract | Single | Dual |
|---|---|---|
| Cameras supplied to StarVLA, in order | context, interaction, wrist_right | context, wrist_left, wrist_right |
| Raw LeRobot state | 111D | 222D |
| Selected state | right EE pose 7 + hand qpos 24 = 31D | left then right, 62D |
| Physical action | EE delta 6 + absolute hand qpos 24 = 30D | left then right, 60D |
| Timing | 30 Hz, 50-action prediction chunk | 30 Hz, 50-action prediction chunk |

The dataset view slices the state without copying parquet or video. Its
`meta/modality.json` maps the source arrays and preserves physical semantic
actions. The evaluation client sends uint8 224×224 RGB and the same compact
state order. The client requests the server's training-time state transform;
`PolicyNormProcessor` also applies the saved action transform in reverse, so
the returned actions are physical. Evo-DexBench's
`encode_action` handles the simulator's native normalized controller input.

## Data and training

Use separate data roots and run IDs for StarVLA. For each embodiment, prepare
a metadata-only view of one training dataset. The source is read only; the
view contains links and a provenance file. Inspect `meta/starvla_view.json`
and the original `meta/dex_benchmark.json` before training. Training episodes
with an `evaluation_v1` range are rejected. Keep the complete source manifest
and any held-out evaluation episodes out of the training mixture.

```bash
export STARVLA_ROOT=/absolute/path/to/starVLA
export EVODEX_ROOT=/absolute/path/to/Evo-DexBench
export DATA_ROOT=/absolute/path/to/new/starvla-evodex-data
cd "$STARVLA_ROOT"
python examples/simBenchmarks/EvoDexBench/train_files/prepare_dataset_view.py \
  --source /absolute/path/to/evodex/single-lerobot-v3 \
  --data-root "$DATA_ROOT" --embodiment single
python examples/simBenchmarks/EvoDexBench/train_files/prepare_dataset_view.py \
  --source /absolute/path/to/evodex/dual-lerobot-v3 \
  --data-root "$DATA_ROOT" --embodiment dual
python examples/simBenchmarks/EvoDexBench/train_files/preflight_data.py \
  --data-root "$DATA_ROOT" --embodiment single
python examples/simBenchmarks/EvoDexBench/train_files/preflight_data.py \
  --data-root "$DATA_ROOT" --embodiment dual
python examples/simBenchmarks/EvoDexBench/train_files/preflight_processor.py \
  --data-root "$DATA_ROOT" --embodiment single
python examples/simBenchmarks/EvoDexBench/train_files/preflight_processor.py \
  --data-root "$DATA_ROOT" --embodiment dual
```

The initial continuous-action baseline uses StarVLA QwenOFT, the same trainer
and q99 transform family used by RoboDojo. The two reference configs define
the action and state sizes; a training run is required before they can be
treated as evaluated baselines. For a one-step training preflight:

```bash
cd "$STARVLA_ROOT"
export WANDB_MODE=disabled
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/EvoDexBench/train_files/starvla_evodex_single_qwenoft_h50_q99.yaml \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --trainer.max_train_steps 1 \
  --run_root_dir /absolute/path/to/new/starvla-runs \
  --run_id evodex-single-preflight-001
```

Replace `single` with `dual` and use a distinct run ID for the second recipe.
Before a real training run, choose a full step budget and separate run ID,
record data and code provenance, and verify that a real sample reaches
`framework.forward()` with finite loss. The default one-step command is a
preflight, never a reported score.

## Two-process evaluation

Before serving a checkpoint, check one simulator reset and physical action
encoding in the Evo-DexBench environment. This uses a neutral action and is
only a simulator diagnostic:

```bash
export PYTHONPATH="$STARVLA_ROOT:$EVODEX_ROOT"
python "$STARVLA_ROOT/examples/simBenchmarks/EvoDexBench/eval_files/preflight_sim.py" \
  --task-id m01_v1 --embodiment single
python "$STARVLA_ROOT/examples/simBenchmarks/EvoDexBench/eval_files/preflight_sim.py" \
  --task-id b11_v1 --embodiment dual
```

On a headless host with a local Mesa Vulkan ICD, set `VK_ICD_FILENAMES` to
its `lvp_icd.x86_64.json` and append `--cpu-render`. This selects the
benchmark's CPU simulation and rendering options for the diagnostic run.

Start the model server in the StarVLA environment. It returns already
unnormalized physical actions. In a second terminal, activate the
Evo-DexBench simulator environment while keeping StarVLA importable, then run
Evo-DexBench's unmodified fixed evaluator through the thin registry wrapper.
The checkpoint must be trained with the matching single/dual recipe.

```bash
# Terminal 1: StarVLA policy environment
cd "$STARVLA_ROOT"
python deployment/model_server/server_policy.py \
  --ckpt_path /absolute/path/to/new/starvla-checkpoint --port 10093 --use_bf16

# Terminal 2: Evo-DexBench simulator environment
export EVODEX_ROOT=/absolute/path/to/Evo-DexBench
export PYTHONPATH="$STARVLA_ROOT:$EVODEX_ROOT"
cd "$STARVLA_ROOT"
python -m examples.simBenchmarks.EvoDexBench.eval_files.run_protocol \
  --policy starvla --task-id m01_v1 \
  --control-mode arm_pd_ee_delta_pose_hand_pd_joint_pos --obs-mode rgb \
  --policy-kwargs '{"host":"127.0.0.1","port":10093,"embodiment":"single","execute_steps":10}' \
  --log /absolute/path/to/new/starvla-runs/eval-m01-001/episodes.jsonl
```

For a **diagnostic** reset and small number of episodes, append `--episodes 2
--num-envs 1 --action-bounds-policy error`. Do not publish that output as a
standard result. The formal command omits these overrides and reads
`configs/eval/protocol_v1.yaml`: 20 episodes each for nominal, spatial,
camera, lighting, and physical (100 total), three task YAML layouts, seeds
1001–1005, task YAML maximum steps, and the frozen `clip` bounds policy.
Use `--task-id b11_v1` and `"embodiment":"dual"` for a dual-arm checkpoint.
The benchmark JSONL records each episode and its summary. Verify completion:

```bash
python examples/simBenchmarks/EvoDexBench/eval_files/verify_log.py \
  /absolute/path/to/new/starvla-runs/eval-m01-001/episodes.jsonl
```

Report Total SR, Stage SR, and D1–D5 only from complete, protocol-valid
task suites and the benchmark's own aggregation. No StarVLA score is supplied
by this code-only integration.

Record the source and checkpoint identities alongside each run:

```bash
python examples/simBenchmarks/EvoDexBench/record_provenance.py \
  --evodex-root "$EVODEX_ROOT" --data-view "$DATA_ROOT/evodex_single" \
  --checkpoint /absolute/path/to/new/starvla-checkpoint \
  --run-id evodex-m01-eval-001 \
  --output /absolute/path/to/new/starvla-runs/eval-m01-001/provenance.json
```

## Failure checks

- A missing camera, non-224 RGB frame, 30 Hz mismatch, state/action dimension
  mismatch, or wrong `unnorm_key` fails before or at the first inference.
- If StarVLA cannot find `meta/modality.json`, point `data_root_dir` to the
  prepared view root; do not edit the source dataset.
- If policy and simulator dependencies conflict, use separate environments and
  the existing WebSocket connection.
- If an evaluation exits early, retain its JSONL as a preflight/error log.
  `verify_log.py` must pass before any number is called a standard result.
