# Morphology-Generalizable Robotic Control via GNN + LLM Planning

> **Zero-shot transfer of a locomotion policy from a 12-DOF quadruped to an 18-DOF hexapod — no retraining, no fine-tuning.**

A research project combining Graph Neural Networks (GNN), Proximal Policy Optimization (PPO), and Large Language Model (LLM) planning to create robot locomotion policies that generalize across different robot morphologies at inference time.

Targeting **ICRA / CoRL** workshop tracks.

---

## Table of Contents

- [Key Result](#key-result)
- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
  - [1. URDF → Graph Conversion](#1-urdf--graph-conversion)
  - [2. GNN Actor-Critic (SlimHeteroGNNActorCritic)](#2-gnn-actor-critic-slimheterognnactorcritic)
  - [3. PyBullet Training Environment](#3-pybullet-training-environment)
  - [4. PPO Training Pipeline](#4-ppo-training-pipeline)
  - [5. Zero-Shot Morphology Transfer](#5-zero-shot-morphology-transfer)
  - [6. LLM Planning Layer](#6-llm-planning-layer)
  - [7. ROS2 / Gazebo Deployment Stack](#7-ros2--gazebo-deployment-stack)
- [MLP Baseline & Why It Fails](#mlp-baseline--why-it-fails)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Train the GNN Policy](#train-the-gnn-policy)
  - [Run LLM-Guided Policy (PyBullet)](#run-llm-guided-policy-pybullet)
  - [Zero-Shot Hexapod Transfer](#zero-shot-hexapod-transfer)
  - [Generate Hexapod URDF](#generate-hexapod-urdf)
  - [Inspect a Checkpoint](#inspect-a-checkpoint)
  - [ROS2 Launch (Gazebo)](#ros2-launch-gazebo)
- [Hyperparameters](#hyperparameters)
- [Observation & Action Spaces](#observation--action-spaces)
- [Reward Function](#reward-function)
- [Results & Plots](#results--plots)
- [What Is Tracked by Git](#what-is-tracked-by-git)
- [Research Status](#research-status)

---

## Key Result

| Transfer | Method | Result |
|---|---|---|
| Quadruped (12 DOF) → Hexapod (18 DOF) | **GNN (ours)** | ✅ Zero-shot — same weights, locomotion achieved |
| Quadruped (12 DOF) → Hexapod (18 DOF) | MLP baseline | ❌ Hard failure — shape mismatch, cannot even load |

The GNN policy's relational inductive bias means the same 29,572 parameters work on any robot whose morphology can be expressed as a graph — quadruped, hexapod, or beyond — without any retraining.

---

## Architecture Overview

```
Natural Language Command
        │
        ▼
┌─────────────────────┐
│   LLM Planner       │  Llama 3.1 8B / Qwen 2.5 7B (via Ollama)
│  (llm_planner_node) │  Outputs JSON: { skill, target, params }
└─────────┬───────────┘
          │ /llm_action  (ROS2 topic)
          ▼
┌─────────────────────┐
│  Skill Translator   │  Maps skill + scene graph → goal_pose
│ (skill_translator)  │
└─────────┬───────────┘
          │ /goal_pose
          ▼
┌──────────────────────────────────────────┐
│         GNN Policy Node                  │
│  • Parses URDF → kinematic graph         │
│  • Builds per-step PyG Data object       │
│  • Runs SlimHeteroGNNActorCritic @ 200Hz │
│  • Publishes joint position commands     │
└──────────────────────────────────────────┘
          │ /model/robot/joint/{name}/cmd_pos
          ▼
┌─────────────────────┐
│  Robot Sim / HW     │  PyBullet (training) | Gazebo + ROS2 (deployment)
└─────────────────────┘
```

---

## How It Works

### 1. URDF → Graph Conversion

**File:** `morpho_gnn_robot/Training_Location/urdf_to_graph.py`

The `URDFGraphBuilder` parses the robot's URDF file and constructs a `torch_geometric.data.Data` graph:

- **Nodes:** One node per controllable joint + one global **body node**
  - Quadruped: 13 nodes (1 body + 12 joints)
  - Hexapod: 19 nodes (1 body + 18 joints)
- **Edges:** Bidirectional kinematic edges between parent–child joints, plus body↔root-joint edges. Edge features encode the 3D joint origin offset + a direction flag.
- **Node roles (heterogeneous):** Each node is typed by its functional role:
  - `0` = body (global state aggregator)
  - `1` = HAA (hip abduction/adduction)
  - `2` = HFE (hip flexion/extension)
  - `3` = KFE (knee flexion/extension)
  - `4` = generic
- **Node features (28-dim):** 11 static URDF features (joint type one-hot, axis, limits) + 17 runtime features (joint pos, joint vel, body quat, gravity vector, linear vel, angular vel, command)

Because the graph topology is read directly from the URDF at runtime, the same code handles any robot with any number of joints — no hardcoding.

**Quadruped node role sequence:** `[body, HAA, HFE, KFE] × 4 legs`
**Hexapod node role sequence:** `[body, HAA, HFE, KFE] × 6 legs`

---

### 2. GNN Actor-Critic (SlimHeteroGNNActorCritic)

**File:** `morpho_gnn_robot/Training_Location/gnn_actor_critic.py`

**Total parameters: 29,572** (well under a 50K budget).

```
Architecture
────────────────────────────────────────────
type_proj   (5 role-specific Linear layers)   5 × (28→48)   = 7,200 params
conv1       GATv2Conv(48→48, heads=2)                        = 18,944 params
norm1       LayerNorm(96)                                     =    192 params
conv2       GATv2Conv(96→48, heads=1)                        =  3,024 params  (approx)
norm2       LayerNorm(48)                                     =     96 params
actor_head  Linear(48→32) → Tanh → Linear(32→1)             =  1,569 params
log_std     learnable (one per joint, initialized at -1.6)   =     12 params  (quadruped)
critic_head Linear(48→32) → Tanh → Linear(32→1)             =  1,569 params
────────────────────────────────────────────
TOTAL                                                         ≈ 29,572
```

**Actor:** Per-joint embeddings from the final GATv2 layer are fed individually through `actor_head` (shared weights) → one scalar action per joint. This is what enables transfer — the head sees a 48-dim embedding, not a flat fixed-size vector.

**Critic:** Global mean pool over all node embeddings → `critic_head` → scalar value estimate.

**Why GATv2?** Graph Attention Networks v2 compute dynamic, input-dependent attention weights on edges, allowing the policy to learn which neighboring joints matter more depending on the current robot state.

---

### 3. PyBullet Training Environment

**File:** `morpho_gnn_robot/Training_Location/robot_env_bullet.py`

A `gymnasium.Env` wrapping PyBullet physics:

| Property | Value |
|---|---|
| Physics timestep | 1/400 s |
| Control frequency | 200 Hz (2 physics steps per action) |
| Action space | `[-1, 1]^N` (N = number of joints) |
| Observation space | `R^(2N + 15)` |
| Joint control | PD position control: Kp=150, Kd=5 |
| Action scale | 0.6 rad around nominal pose |
| Action smoothing | α=0.8 exponential moving average |

**Termination conditions:**
- Base height < 0.25 m (fall) — synced with MLP baseline
- |roll| or |pitch| > 0.8 rad (tilt)
- Non-foot link contact with ground

**Fall penalty:** 250.0 (synced with MLP baseline — strong signal to avoid falls rather than tolerate them)

**Command-conditioned training:** At each episode reset, a random command mode is sampled:
- Stand still: `[vx=0, wy=0]`
- Move forward: `[vx ~ U(0.5, 1.0), wy=0]`
- Rotate: `[vx=0, wy ~ ±U(0.5, 1.0)]`

The command vector is broadcast to all joint nodes and the body node in the graph.

**Domain randomization:** Gaussian observation noise (σ=0.01 pos, σ=0.02 vel), random initial orientation perturbation (±0.05 rad).

---

### 4. PPO Training Pipeline

**File:** `morpho_gnn_robot/Training_Location/train_gnn_ppo.py`

Standard CleanRL-style PPO with graph-batched rollouts via `torch_geometric.data.Batch`.

| Hyperparameter | Value |
|---|---|
| Total timesteps | 12,000,000 |
| Rollout steps per update | 4,096 |
| Minibatch size | 1,024 (4 minibatches) |
| Update epochs | 6 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| PPO clip ε | 0.15 |
| Entropy coef | 0.005 (fixed) |
| Value loss coef | 0.5 |
| Clip value loss | True |
| Max grad norm | 0.5 |
| Target KL (early stop) | 0.015 |
| Learning rate (GNN trunk) | 6.8e-4 |
| Learning rate (actor head) | 4.5e-4 |
| Learning rate (critic head) | 4.5e-4 |
| LR schedule | Linear decay to 0 |
| Hidden dim | 48 |

**Separate parameter groups:** GNN layers, actor head, and critic head each have their own Adam optimizer group, allowing independent LR scheduling.

**Running observation normalization:** Online mean/variance normalization (Welford-style) over the first 30 dimensions is saved into every checkpoint for consistent inference.

**Checkpoint format:**
```python
{
    'global_step': int,
    'agent': state_dict,
    'optimizer': optimizer_state_dict,
    'episode_rewards': List[float],
    'obs_norm_mean': np.ndarray,
    'obs_norm_var': np.ndarray,
    'obs_norm_count': float,
}
```

Optional W&B logging with `--track 1`.

---

### 5. Zero-Shot Morphology Transfer

**File:** `morpho_gnn_robot/Training_Location/test_morphology_transfer.py`

The transfer procedure loads a quadruped-trained checkpoint onto an 18-joint hexapod model:

1. **Graph rebuild:** `URDFGraphBuilder` parses `hexapod_anymal.urdf` → 19-node graph (1 body + 18 joints)
2. **Weight reuse:** All GNN layers (`type_proj`, `conv1`, `conv2`) and the shared `actor_head` / `critic_head` load directly — their shapes are independent of the number of joints.
3. **`log_std` expansion:** The only morphology-dependent parameter. The 12-dim quadruped `log_std` is expanded to 18-dim by replicating the mean value for the 6 new joints.
4. **Observation normalization adaptation:** The 30-dim running norm statistics from quadruped training are remapped to the 42-dim hexapod observation by duplicating leg-group statistics for the new middle legs.

The hexapod URDF is generated programmatically from the quadruped URDF:

**File:** `morpho_gnn_robot/Training_Location/generate_hexapod.py`

Clones the `LF_*` and `RF_*` leg kinematic chains, renames them `LM_*` / `RM_*`, and offsets their hip origins by -0.277 m along X to create a physically valid 6-legged body.

---

### 6. LLM Planning Layer

**File (standalone):** `morpho_gnn_robot/Training_Location/run_llm_policy.py`

A single-process runner that:
1. Accepts a natural language instruction (`--instruction "move forward"`)
2. Maps the instruction to a velocity command `[vx, wy]`
3. Runs the GNN policy in PyBullet GUI with that command override

**File (ROS2):** `morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/morpho_robot/llm_planner_node.py`

A ROS2 node (`llm_planner_node`) with a **two-tier planning architecture**:
- **Fast reactive path (0.5 s interval):** Pure rule-based `reactive_fallback()` using depth sensor readings. Selects `trot`, `turn_left`, or `turn_right` based on front/side distances. Includes a `SIDE_WALL_MIN=1.5m` guard to prevent turning into a close side wall (which causes roll instability).
- **Slow LLM path (5.0 s interval):** Calls Ollama (Qwen 2.5 7B default) and publishes structured JSON plan to `/llm_action`: `{ skill, target, params }`. Falls back gracefully to reactive mode when Ollama is unavailable.

**File:** `morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/morpho_robot/vision_node.py`

YOLOv8 + depth camera perception node:
- Publishes scene graph JSON to `/scene_graph` with detected objects and obstacle distances
- **Ground-filtered depth ROI:** Only reads image rows 10%–55% from top to exclude ground/floor pixels when the robot pitches forward. Prevents false obstacle triggers from floor detections.

**File:** `morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/morpho_robot/skill_translator_node.py`

The `skill_translator_node` bridges high-level plans to navigation goals:
- Parses the LLM JSON action
- Resolves the `target` against the scene graph (object label + distance + bearing)
- Converts to a `geometry_msgs/PoseStamped` on `/goal_pose`

---

### 7. ROS2 / Gazebo Deployment Stack

Two policy nodes are available, selected via the `policy_type` launch argument:

**`MLP_policy_node.py`** *(currently deployed — stable locomotion confirmed)*
- Runs the MLP baseline policy (`mlp_ppo_10223616.pt`) at **200 Hz**
- Differential steering via `steer_bias` (±0.08) applied to left/right leg action scaling
- Telemetry: pitch, roll (corrected formula: `arctan2(y, -z)`), and L/R action magnitude
- Roll warning threshold: 15°

**`gnn_policy_node.py`** *(in development — GNN retraining in progress)*
- Runs `SlimHeteroGNNActorCritic` at **200 Hz**
- Subscribes to `/joint_states` and `/odom`
- Converts world-frame velocities to body-frame using quaternion rotation
- 400-tick linear ramp-up on startup to avoid joint impulses
- Yaw-rate PD correction on HAA joints for heading stability

**Common infrastructure:**
- `parameter_bridge` (gz-ros2): bridges Gazebo topics ↔ ROS2
- `warehouse_world.sdf`: Gazebo world (robot spawned separately via `spawn_entity` — no embedded URDF include to prevent ghost-model conflicts)
- Publishes per-joint `Float64` commands on `/model/robot/joint/{name}/cmd_pos`

---

## MLP Baseline & Why It Fails

**Files:** `morpho_gnn_robot/Training_MLP/`

A standard MLP (256-256 hidden dims, ~200K parameters) is trained on the same quadruped environment with the same PPO hyperparameters. When transfer to the hexapod is attempted:

```
❌ TRANSFER FAILED!
size mismatch for trunk.0.weight: copying a param with shape
torch.Size([256, 37]) from checkpoint, the shape in current
model is torch.Size([256, 49]).
```

**Root cause:** The MLP's first layer fuses all 37 observation dims into a fixed-width hidden layer. The hexapod produces 49 observations (12 more joint signals). There is no structural concept of "joints" or "edges" — just a hardcoded flat array. Morphology transfer requires deleting all weights and retraining from scratch.

The GNN has no such constraint: joint embeddings are computed per-node through shared message-passing operations. Adding 6 more joint nodes to the graph is transparent to the learned weights.

---

## Repository Structure

```
.
├── morpho_gnn_robot/
│   ├── Training_Location/           # Core RL training & transfer code
│   │   ├── gnn_actor_critic.py      # SlimHeteroGNNActorCritic (29,572 params)
│   │   ├── train_gnn_ppo.py         # PPO training loop (2M steps)
│   │   ├── robot_env_bullet.py      # PyBullet Gym environment
│   │   ├── urdf_to_graph.py         # URDF → PyTorch Geometric graph
│   │   ├── run_llm_policy.py        # LLM command → GNN policy (standalone)
│   │   ├── test_morphology_transfer.py  # Zero-shot quad→hex transfer demo
│   │   ├── generate_hexapod.py      # Procedural hexapod URDF generator
│   │   ├── hexapod_anymal.urdf      # Generated 18-DOF hexapod URDF
│   │   ├── anymal_stripped.urdf     # Cleaned 12-DOF quadruped URDF
│   │   ├── stripped_urdf_maker.py   # URDF cleaning utility
│   │   ├── check_checkpoint.py      # Value function variance inspector
│   │   ├── smoke_test.py            # Quick environment sanity check
│   │   ├── test_action.py           # Action space validation
│   │   ├── test_critic.py           # Critic output tests
│   │   ├── test_ev.py               # Explained variance diagnostic
│   │   ├── test_stand.py            # Static standing test
│   │   └── test_vf_variance.py      # Value function variance test
│   │
│   ├── Training_MLP/                # MLP baseline (demonstrates transfer failure)
│   │   ├── mlp_actor_critic.py      # Standard MLP policy (~200K params)
│   │   ├── train_mlp_ppo.py         # MLP PPO training
│   │   ├── robot_env_bullet.py      # Same environment
│   │   ├── test_mlp_transfer_failure.py  # Proves MLP cannot transfer
│   │   ├── anymal_stripped.urdf
│   │   └── hexapod_anymal.urdf
│   │
│   ├── plots/                       # Publication-ready figures
│   │   ├── zero_shot_transfer_barplot.png
│   │   ├── zero_shot_transfer_barplot.pdf
│   │   ├── zero_shot_transfer_boxplot.png
│   │   └── zero_shot_transfer_boxplot.pdf
│   │
│   └── morpho_ros2_ws/              # ROS2 workspace (Gazebo deployment)
│       └── src/morpho_robot/
│           ├── morpho_robot/
│           │   ├── gnn_policy_node.py       # 200Hz GNN control loop
│           │   ├── llm_planner_node.py      # Ollama LLM → /llm_action
│           │   ├── skill_translator_node.py # Plan → /goal_pose
│           │   ├── MLP_policy_node.py       # MLP policy node (baseline)
│           │   ├── vision_node.py           # Camera perception
│           │   ├── read_joints.py           # Joint state reader
│           │   ├── gnn_actor_critic.py      # GNN model (copy for ROS)
│           │   └── urdf_to_graph.py         # Graph builder (copy for ROS)
│           ├── launch/morpho_robot.launch.py
│           ├── config/bridge.yaml
│           ├── urdf/anymal.urdf
│           ├── worlds/warehouse_world.sdf
│           └── meshes/                      # ANYmal visual meshes
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

### Dependencies

```bash
pip install torch torchvision
pip install torch-geometric
pip install pybullet
pip install gymnasium
pip install scipy numpy

ollama pull llama3.1:8b
ollama pull qwen2.5:7b

pip install wandb
```

### ROS2 Deployment (additional)

```bash
sudo apt install ros-jazzy-desktop

cd morpho_gnn_robot/morpho_ros2_ws
colcon build --symlink-install
source install/setup.bash

pip install ollama rclpy
```

---

## Usage

### Train the GNN Policy

```bash
cd morpho_gnn_robot/Training_Location

# Default run (12M steps, MLP-aligned hyperparameters)
python train_gnn_ppo.py

# Custom run
python train_gnn_ppo.py \
  --total-timesteps 12000000 \
  --gnn-learning-rate 0.00068 \
  --actor-learning-rate 0.00045 \
  --critic-learning-rate 0.00045 \
  --num-steps 4096 \
  --clip-coef 0.15 \
  --hidden-dim 48 \
  --seed 0 \
  --checkpoint-dir ./checkpoints \
  --save-every 70000

# Resume from checkpoint
python train_gnn_ppo.py \
  --resume-path ./checkpoints/gnn_ppo_1400000.pt

# Resume without restoring optimizer state (fresh LR)
python train_gnn_ppo.py \
  --resume-path ./checkpoints/gnn_ppo_1400000.pt \
  --resume-optimizer 0

# With W&B logging
python train_gnn_ppo.py --track 1 --run-name my_run
```

Checkpoints are saved every 70,000 steps and as `gnn_ppo_final.pt` at the end.

---

### Run LLM-Guided Policy (PyBullet)

```bash
cd morpho_gnn_robot/Training_Location

python run_llm_policy.py \
  --checkpoint ./checkpoints/gnn_ppo_final.pt \
  --urdf anymal_stripped.urdf \
  --instruction "move forward"

python run_llm_policy.py \
  --checkpoint ./checkpoints/gnn_ppo_final.pt \
  --instruction "turn left" \
  --stochastic
```

PyBullet GUI opens automatically. The policy receives the LLM-mapped velocity command `[vx, wy]` and runs indefinitely until `Ctrl+C`.

---

### Zero-Shot Hexapod Transfer

```bash
cd morpho_gnn_robot/Training_Location

python test_morphology_transfer.py

python test_morphology_transfer.py ./checkpoints/gnn_ppo_final.pt
```

This loads the quadruped-trained weights, expands `log_std` from 12→18, remaps running norm statistics, and runs the hexapod in PyBullet GUI in real time.

---

### Generate Hexapod URDF

```bash
cd morpho_gnn_robot/Training_Location
python generate_hexapod.py
```

---

### Inspect a Checkpoint

```bash
cd morpho_gnn_robot/Training_Location
python check_checkpoint.py
```

---

### Demonstrate MLP Transfer Failure

```bash
cd morpho_gnn_robot/Training_MLP
python test_mlp_transfer_failure.py
```

---

### ROS2 Launch (Gazebo)

```bash
cd morpho_gnn_robot/morpho_ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select morpho_robot --symlink-install
source install/setup.bash

# Launch with MLP policy (stable — recommended)
ros2 launch morpho_robot morpho_robot.launch.py \
  policy_type:=mlp \
  mlp_checkpoint:=/path/to/mlp_ppo_10223616.pt

# Launch with GNN policy
ros2 launch morpho_robot morpho_robot.launch.py \
  policy_type:=gnn \
  gnn_checkpoint:=/path/to/gnn_ppo_final.pt
```

The launch file starts: `robot_state_publisher`, `gz sim` (warehouse world), `parameter_bridge`, `vision_node` (YOLOv8 + depth), `MLP_policy_node` or `gnn_policy_node`, `llm_planner_node`, and `skill_translator_node`.

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Total timesteps | 12,000,000 | Matched to MLP baseline |
| `num_steps` | 4,096 | Steps per rollout |
| `num_minibatches` | 4 | → minibatch size 1024 |
| `update_epochs` | 6 | Epochs per PPO update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE bias-variance trade-off |
| `clip_coef` | 0.15 | PPO clipping (matched to MLP) |
| `clip_vloss` | True | Clipped value loss (matched to MLP) |
| `ent_coef` | 0.005 | Fixed entropy coefficient |
| `vf_coef` | 0.5 | Value loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `target_kl` | 0.015 | KL early-stop threshold |
| `gnn_learning_rate` | 6.8e-4 | GNN trunk (matched to MLP) |
| `actor_learning_rate` | 4.5e-4 | Actor head |
| `critic_learning_rate` | 4.5e-4 | Critic head |
| LR schedule | Linear decay | Frac from 1.0 → 0.0 over training |
| `hidden_dim` | 48 | GNN hidden width |
| `max_episode_steps` | 1,000 | ~2.5 seconds at 400 Hz |
| `save_every` | 70,000 | Checkpoint interval |

---

## Observation & Action Spaces

### Quadruped Observation (39-dim)

| Slice | Dims | Content |
|---|---|---|
| `obs[0:12]` | 12 | Joint positions (rad) |
| `obs[12:24]` | 12 | Joint velocities (rad/s) |
| `obs[24:27]` | 3 | Body linear velocity (body frame) |
| `obs[27:30]` | 3 | Body angular velocity (body frame) |
| `obs[30:34]` | 4 | Body orientation quaternion (yaw-zeroed) |
| `obs[34:37]` | 3 | Gravity vector in body frame |
| `obs[37:39]` | 2 | Velocity command `[vx, wy]` |

First 30 dims are running-normalized during training. The last 9 dims are raw.

### Action Space

`[-1, 1]^N` continuous, scaled by 0.6 rad around nominal joint poses. Action smoothing applied (α=0.8 EMA).

---

## Reward Function

```
r = r_alive + r_vel + r_stability + r_torque + r_contact

r_alive     =  1.0  (or -100.0 on fall)
r_vel       =  1.5 * exp(-2 * (vx - cmd_vx)²)
             + 1.5 * exp(-2 * (wy - cmd_wy)²)
r_stability = -1.0 * (roll² + pitch²)
             - 0.5 * |lateral_vel|
r_torque    = -5e-6 * sum(torques²)
r_contact   = -0.5 if non-foot link touches ground (+ terminates)

clipped to [-5.0, 5.0]
```

---

## Results & Plots

Publication-ready plots are in `morpho_gnn_robot/plots/`:

| File | Description |
|---|---|
| `zero_shot_transfer_barplot.png/.pdf` | Bar chart: GNN vs MLP across morphologies |
| `zero_shot_transfer_boxplot.png/.pdf` | Box plot: episode reward distributions |

---

## What Is Tracked by Git

Checkpoints (`.pt`, `.pth`), W&B logs, build artifacts, and large binary weights are **not tracked** (see `.gitignore`).

**Tracked files include:**
- All Python source files
- URDF files (`anymal_stripped.urdf`, `hexapod_anymal.urdf`, `anymal.urdf`)
- ROS2 package config (`package.xml`, `setup.py`, `setup.cfg`, `bridge.yaml`)
- Launch files and world SDF
- ANYmal mesh files (`.dae`, textures)
- Result plots (`plots/*.png`, `plots/*.pdf`)
- `LICENSE`, `README.md`, `.gitignore`

**Not tracked (gitignored):**
- `checkpoints/`, `*.pt`, `*.pth`, `*.onnx` — model weights
- `build/`, `install/`, `log/` — ROS2 colcon artifacts
- `wandb/`, `runs/` — experiment logs
- `.venv/`, `__pycache__/`, `*.pyc`
- `.env`, `*api_key*` — secrets

---

## Research Status

| Component | Status |
|---|---|
| MLP baseline training (12M steps) | ✅ Complete — `mlp_ppo_10223616.pt` |
| MLP transfer failure demo | ✅ Complete |
| GNN policy training (quadruped) | 🔄 Retraining — aligned to MLP hyperparameters |
| Zero-shot hexapod transfer | ✅ Demonstrated in PyBullet |
| Hexapod URDF generator | ✅ Complete |
| LLM planning layer (standalone) | ✅ Complete |
| ROS2 / Gazebo: MLP policy deployment | ✅ Stable locomotion + obstacle avoidance |
| ROS2 / Gazebo: GNN policy deployment | 🔄 Pending GNN retraining completion |
| Vision node (YOLOv8 + depth, ground filter) | ✅ Complete |
| Reactive obstacle avoidance (0.5 s loop) | ✅ Complete — side-wall guard included |
| Roll telemetry fix (`arctan2(y, -z)`) | ✅ Fixed |
| Spawn conflict fix (world SDF) | ✅ Fixed |
| Publication plots | ✅ Generated |
| Paper draft | 🔄 In progress |
| Target venue | ICRA / CoRL workshop |

---

## License

See [LICENSE](./LICENSE).

---

*Research project by Chaitanya Parate — MIT World Peace University, Pune.*
*Conducted as part of an independent research initiative in robot learning and morphological generalization.*
