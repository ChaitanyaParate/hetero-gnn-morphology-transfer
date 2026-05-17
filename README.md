# Constrained Zero-Shot Morphological Transfer for Legged Locomotion via Heterogeneous Graph Neural Networks

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20187567-blue.svg)](https://doi.org/10.5281/zenodo.20187567)

> **Zero-shot transfer of a locomotion policy from a 12-DOF quadruped to an 18-DOF hexapod — no retraining required. 500K-step fine-tuning yields a 3.8× reward gain.**

A research project combining Graph Neural Networks (GNN), Proximal Policy Optimization (PPO), and Large Language Model (LLM) planning to create robot locomotion policies that generalize across different robot morphologies at inference time.



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
- [🖥️ Test Environment](#️-test-environment)
- [Installation & Running](#installation--running)
  - [Option A — Docker (Recommended)](#option-a--docker-recommended)
  - [Option B — Manual Install](#option-b--manual-install)
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

| Experiment | Method | Reward | Survival |
|---|---|---|---|
| Quadruped in-distribution | GNN (31,582 params) | 2,409 ± 575 | 1,000 steps (100%) |
| Quadruped in-distribution | MLP (210,457 params) | 4,237 ± 343 | 1,000 steps (100%) |
| Hexapod zero-shot (18 DOF) | **GNN — same weights** | 106 ± 25 | ~47 steps |
| Hexapod zero-shot (18 DOF) | MLP | ❌ RuntimeError | 0 steps |
| Hexapod fine-tuned (500K steps) | GNN | **416 ± 114** | **193 steps (+3.8×)** |
| Aliengo/Go1 zero-shot (12 DOF) | GNN | 112–526 | 60–184 steps |
| Aliengo/Go1 zero-shot (12 DOF) | MLP | ❌ RuntimeError | 0 steps |
| Terrain 5° slope (zero-shot) | GNN | 2,440 ± 780 | **95% success** |
| Terrain 10° slope (zero-shot) | GNN | 187 ± 33 | 0% success |

> **Both GNN and MLP use identical per-step reward clipped to [−5, 5].** MLP's higher reward reflects 6.7× more parameters (85% vs 50% of theoretical maximum) — not a different reward function. The GNN trades peak in-distribution reward for zero-shot generalization.

The MLP cannot load on *any* morphology other than the training one (fixed input dimension). The GNN runs on every morphology in the HAA/HFE/KFE vocabulary without code change.

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

**File:** `morpho_gnn_robot/core/urdf_to_graph.py`

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

**File:** `morpho_gnn_robot/core/gnn_actor_critic.py`

**Total parameters: 31,582** (85% fewer than the MLP baseline's 210,457).

```
Architecture
────────────────────────────────────────────
type_proj   (5 role-specific Linear layers)   5 × (28→48)   = 7,200 params
conv1       GATv2Conv(48→48, heads=2, edge_dim=4)            = 19,680 params
norm1       LayerNorm(96)                                     =    192 params
conv2       GATv2Conv(96→48, heads=1, edge_dim=4)            =  1,296 params
norm2       LayerNorm(48)                                     =     96 params
actor_head  Linear(48→32) → Tanh → Linear(32→1)             =  1,569 params
log_std     learnable (one per joint, initialized at -1.6)   =     12 params  (quadruped)
critic_head Linear(96→32) → ELU  → Linear(32→1)             =  1,569 params  (dual-stream)
────────────────────────────────────────────
TOTAL                                                         ≈ 31,582
```

> **Node features are 28-dim:** 11 static URDF features + 17 runtime features (joint pos, vel, body lin/ang vel, quat, gravity, command). Body angular velocity was added to the static embedding in the final architecture revision.

**Actor:** Per-joint embeddings from the final GATv2 layer are fed individually through `actor_head` (shared weights) → one scalar action per joint. This is what enables transfer — the head sees a 48-dim embedding, not a flat fixed-size vector.

**Critic:** Global mean pool over all node embeddings → `critic_head` → scalar value estimate.

**Why GATv2?** Graph Attention Networks v2 compute dynamic, input-dependent attention weights on edges, allowing the policy to learn which neighboring joints matter more depending on the current robot state.

---

### 3. PyBullet Training Environment

**File:** `morpho_gnn_robot/core/robot_env_bullet.py`

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

**Command-conditioned training:** Training is locked to **forward-walking only** to break the crouching local optimum:
- Move forward: `[vx ~ U(0.5, 1.0), wy=0]` — always, every episode

The command vector is broadcast to all joint nodes and the body node in the graph. This means `wy` is **always 0.0** during training — an important constraint to respect at deployment time.

**Domain randomization:** Gaussian observation noise (σ=0.01 pos, σ=0.02 vel), random initial orientation perturbation (±0.05 rad).

---

### 4. PPO Training Pipeline

**File:** `morpho_gnn_robot/Training_GNN/train_gnn_ppo.py`

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

**File:** `morpho_gnn_robot/Training_GNN/test_morphology_transfer.py`

The transfer procedure loads a quadruped-trained checkpoint onto an 18-joint hexapod model:

1. **Graph rebuild:** `URDFGraphBuilder` parses `hexapod_anymal.urdf` → 19-node graph (1 body + 18 joints)
2. **Weight reuse:** All GNN layers (`type_proj`, `conv1`, `conv2`) and the shared `actor_head` / `critic_head` load directly — their shapes are independent of the number of joints.
3. **`log_std` expansion:** The only morphology-dependent parameter. The 12-dim quadruped `log_std` is expanded to 18-dim by replicating the mean value for the 6 new joints.
4. **Observation normalization adaptation:** The 30-dim running norm statistics from quadruped training are remapped to the 42-dim hexapod observation by duplicating leg-group statistics for the new middle legs.

The hexapod URDF is generated programmatically from the quadruped URDF:

**File:** `morpho_gnn_robot/Training_GNN/generate_hexapod.py`

Clones the `LF_*` and `RF_*` leg kinematic chains, renames them `LM_*` / `RM_*`, and offsets their hip origins by -0.277 m along X to create a physically valid 6-legged body.

---

### 6. LLM Planning Layer

**File (standalone):** `morpho_gnn_robot/Training_GNN/run_llm_policy.py`

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
- **Ground-filtered depth ROI:** Only reads the top 5%–40% of image rows — floor pixels only appear in lower rows. Using the lower 55% caused the floor to be detected as the closest obstacle, triggering constant turn commands.
- **Front distance filtering:** The centre pixel is checked against `MIN_OBSTACLE_DIST=2.0m` before use. If the centre ray hits the floor or a leg, a small horizontal band is scanned for a real obstacle reading.
- **Robust closest-obstacle:** 5th-percentile of the middle-column ROI (not raw minimum) rejects isolated floor/leg pixels.

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

**`gnn_policy_node.py`** *(active — stable forward locomotion confirmed in Gazebo)*
- Runs `SlimHeteroGNNActorCritic` at **200 Hz**
- Subscribes to `/joint_states` and `/odom`
- Converts world-frame velocities to body-frame using quaternion rotation
- 400-tick linear ramp-up on startup to avoid joint impulses
- **In-distribution command clamping:** `vx` clamped to `[0.5, 1.0]` (training range); `wy` clamped to `±0.3` (GNN was trained with `wy=0` only, so large values are OOD)
- **Yaw-rate PI controller:** Post-action HAA (hip abductor) joint correction — `KP=0.35`, `KI=0.02`, `LPF=0.90`. Offsets left/right HAA joints to counteract circular drift without modifying the GNN's input command distribution

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
│   │
│   ├── core/                            # ★ Single source of truth for shared modules
│   │   ├── __init__.py
│   │   ├── gnn_actor_critic.py          # SlimHeteroGNNActorCritic (31,582 params)
│   │   ├── mlp_actor_critic.py          # MLP baseline policy (210,457 params)
│   │   ├── urdf_to_graph.py             # URDF → PyTorch Geometric graph
│   │   └── robot_env_bullet.py          # PyBullet Gym env (stderr suppressor included)
│   │
│   ├── URDFs/                           # ★ Single canonical URDF location
│   │   ├── anymal.urdf                  # Full ANYmal URDF
│   │   ├── anymal_stripped.urdf         # Collision-only quadruped URDF (training)
│   │   ├── hexapod_anymal.urdf          # Generated 18-DOF hexapod URDF
│   │   ├── aliengo.urdf                 # Unitree Aliengo URDF
│   │   ├── aliengo_stripped.urdf        # Collision-only Aliengo (eval)
│   │   ├── go1.urdf                     # Unitree Go1 URDF
│   │   ├── go1_stripped.urdf            # Collision-only Go1 (eval)
│   │   └── solo.srdf
│   │
│   ├── Training_GNN/               # GNN RL training & evaluation scripts
│   │   ├── train_gnn_ppo.py             # PPO training loop (12M steps, 5 seeds)
│   │   ├── finetune_transfer.py         # Staged fine-tuning on target morphologies
│   │   ├── eval_comprehensive.py        # Zero-shot vs fine-tuned benchmark
│   │   ├── eval_third_party_transfer.py # Unitree zero-shot evaluation
│   │   ├── evaluate_policies.py         # General policy evaluation script
│   │   ├── run_llm_policy.py            # LLM command → GNN policy (standalone PyBullet)
│   │   ├── test_morphology_transfer.py  # Zero-shot quad→hex transfer demo
│   │   ├── generate_hexapod.py          # Procedural 18-DOF hexapod URDF generator
│   │   ├── record_video.py              # Supplementary video recorder (4 scenes)
│   │   └── eval_results*.json           # Benchmark result files
│   │
│   ├── Training_MLP/                    # MLP baseline (demonstrates transfer failure)
│   │   ├── train_mlp_ppo.py             # MLP PPO training (12M steps, 5 seeds)
│   │   └── test_mlp_transfer_failure.py
│   │
│   ├── best_model/                      # Top checkpoints from 5-seed sweep
│   │   ├── gnn_ppo_*.pt                 # Best GNN checkpoints per seed
│   │   └── mlp_ppo_*.pt
│   │
│   ├── morpho_ros2_ws/                  # ROS2 Jazzy / Gazebo Harmonic workspace
│   │   └── src/morpho_robot/
│   │       ├── morpho_robot/            # Python package (ROS2 nodes)
│   │       │   ├── gnn_policy_node.py       # 200Hz GNN inference loop
│   │       │   ├── MLP_policy_node.py       # 200Hz MLP inference loop
│   │       │   ├── llm_planner_node.py      # Ollama LLM → /llm_action
│   │       │   ├── skill_translator_node.py # Plan → /goal_pose
│   │       │   └── vision_node.py           # YOLOv8 + depth perception
│   │       ├── urdf/                        # ROS2 colcon URDF copies (required by package)
│   │       │   ├── anymal.urdf
│   │       │   └── anymal_stripped.urdf
│   │       ├── launch/morpho_robot.launch.py
│   │       ├── config/bridge.yaml
│   │       ├── worlds/warehouse_world.sdf
│   │       └── package.xml / setup.py
│   │
│   └── plots/                           # Publication-ready result figures
│       ├── zero_shot_transfer_barplot.png/.pdf
│       └── zero_shot_transfer_boxplot.png/.pdf
│
├── GNN_Fine-tuning_output/              # Fine-tuning checkpoints & learning curves
│   ├── Hexapod/
│   │   ├── finetuned_hexapod_final.pt   # ★ 500K steps → reward 110→416 (+3.8×)
│   │   ├── finetuned_hexapod_*.pt       # Intermediate checkpoints
│   │   └── curve_hexapod.json           # Learning curve data
│   ├── aliengo/
│   │   ├── finetuned_aliengo_final.pt
│   │   └── curve_aliengo.json
│   └── go1/
│       ├── finetuned_go1_final.pt
│       └── curve_go1.json
│
├── kaggle_gnn_finetune/                 # Kaggle fine-tuning notebooks
│   └── (Jupyter notebooks for hexapod / Aliengo / Go1 fine-tuning on Kaggle GPU)
│
├── training_with_different_seed_notebooks/  # 5-seed training notebooks + timing data
│   ├── README.md                        # Seed → checkpoint → reward mapping
│   ├── gnn-training-seed-{0..4}-part-*.ipynb
│   ├── mlp-training-seed-{0..4}-part-*.ipynb
│   └── time_taken_to_train_gnn_and_mlp_per_seed.csv
│
├── installers/                          # OS-specific setup scripts (gitignored)
│   ├── README.md
│   ├── Linux/setup_linux.sh
│   ├── Windows/setup_windows.bat + .ps1
│   └── macOS/setup_macOS.app/
│
├── Dockerfile                           # Container image definition
├── docker-compose.yml                   # Dev-mode container orchestration
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md

# Not tracked (gitignored):
#   no_push/            — paper drafts & submission files
#   research_papers/    — reference PDFs
#   *.pt / *.pth        — model checkpoints
#   llama.cpp/          — LLM inference engine
#   build/ install/ log/ — ROS2 colcon artifacts
#   installers/         — setup scripts (distributed separately with the .tar)
```

---

## 🖥️ Test Environment

This project was developed and all experiments were run on the following hardware:

| Component | Spec |
|---|---|
| **OS** | Ubuntu 24.04.4 LTS |
| **CPU** | Intel Core i5-10300H @ 2.50 GHz (4 cores / 8 threads, up to 4.5 GHz) |
| **GPU** | NVIDIA GeForce GTX 1650 Ti (4 GB VRAM) |
| **CUDA** | 13.2 · Driver 595.58.03 |
| **RAM** | 16 GB |
| **Storage** | NVMe SSD |
| **Python** | 3.12.3 |

> **Note:** The GTX 1650 Ti is a **laptop-class GPU with only 4 GB VRAM**. All training (12M PPO steps for GNN, MLP baseline) and zero-shot evaluation completed successfully on this hardware, demonstrating the project is accessible without high-end compute.

---

## Installation & Running

### Option A — Docker (Recommended)

The entire environment (ROS2 Jazzy, PyTorch, Gazebo Harmonic, all Python deps) is pre-built into a single Docker image. **No manual dependency installation required.**

#### Step 1 — Download the Docker image

The pre-built image (`hetero_gnn_project.tar`, **8.3 GB**) is archived on Zenodo with a permanent DOI:

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20187567-blue.svg)](https://doi.org/10.5281/zenodo.20187567)

**Primary — Zenodo (permanent, citable):**
```bash
wget "https://zenodo.org/records/20187567/files/hetero_gnn_project.tar"
# or
curl -L -o hetero_gnn_project.tar "https://zenodo.org/records/20187567/files/hetero_gnn_project.tar"
```

**Alternative — Google Drive (faster download):**
```bash
pip install gdown
gdown 1tI2VpsGHoFGOhWkKnZxwH05tAVnjoXd5 -O hetero_gnn_project.tar
```

Or download manually: [Google Drive link](https://drive.google.com/file/d/1tI2VpsGHoFGOhWkKnZxwH05tAVnjoXd5/view?usp=sharing)

Once downloaded, place `hetero_gnn_project.tar` into the OS-specific installer folder (see Step 2).

#### Step 2 — Run the setup script for your OS

| Operating System | Installer location | How to run |
|---|---|---|
| 🐧 **Ubuntu / Linux** | `installers/Linux/` | `chmod +x setup_linux.sh && ./setup_linux.sh` |
| 🪟 **Windows 10/11** | `installers/Windows/` | Double-click `setup_windows.bat` |
| 🍎 **macOS** | `installers/macOS/` | Double-click `setup_macOS.app` in Finder |

Each installer folder contains a detailed `README.md` covering prerequisites, step-by-step instructions, and a troubleshooting guide.

> **Hardware note:** An NVIDIA GPU is strictly required for full performance (CUDA + Gazebo hardware rendering). macOS is supported for inspection only — PyTorch falls back to CPU.

#### Step 3 — You're in

The setup script drops you into the ROS2 workspace inside the container:
```
root@...:/workspace/morpho_gnn_robot/morpho_ros2_ws#
```
All ROS2, Python, and Gazebo commands work immediately from here.

#### Building the image yourself (optional)

If you prefer to build from source instead of loading the pre-built `.tar`:
```bash
# Build (takes ~20–30 min; requires NVIDIA GPU + Docker + nvidia-container-toolkit)
docker compose build

# Launch (dev mode — mounts project directory live into /workspace)
docker compose up
```

---

### Option B — Manual Install

For development or if Docker is not available.

#### Python dependencies

```bash
pip install torch torchvision
pip install torch-geometric
pip install pybullet gymnasium scipy numpy wandb

# LLM models (via Ollama)
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

#### ROS2 Deployment (additional)

```bash
# Ubuntu 24.04 only
sudo apt install ros-jazzy-desktop ros-jazzy-ros-gz

cd morpho_gnn_robot/morpho_ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select morpho_robot
source install/setup.bash
```

---

## Usage

### Train the GNN Policy

```bash
cd morpho_gnn_robot/Training_GNN

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

### Fine-Tune on a Target Morphology

Run staged fine-tuning (100K steps head-only → 400K steps full network):

```bash
cd morpho_gnn_robot/Training_GNN

# Fine-tune on hexapod (recommended — same HAA/HFE/KFE vocabulary)
python finetune_transfer.py \
  --morphology hexapod \
  --base-checkpoint ./checkpoints/gnn_ppo_final.pt \
  --total-steps 500000 \
  --stage1-steps 100000 \
  --save-dir ./finetune_out

# Fine-tune on Unitree Aliengo (requires full-inertia URDF for best results)
python finetune_transfer.py \
  --morphology aliengo \
  --base-checkpoint ./checkpoints/gnn_ppo_final.pt
```

**Expected hexapod results:** reward 110 → 416 (+3.8×), survival 47 → 193 steps at 500K steps.

---

### Evaluate Zero-Shot vs Fine-Tuned

```bash
cd morpho_gnn_robot/Training_GNN

# Benchmark all morphologies (zero-shot from base checkpoint)
python eval_comprehensive.py \
  --base-checkpoint ./checkpoints/gnn_ppo_final.pt \
  --episodes 20

# Include fine-tuned checkpoints for comparison
python eval_comprehensive.py \
  --base-checkpoint ./checkpoints/gnn_ppo_final.pt \
  --finetuned-dir ./finetune_out \
  --episodes 20
```

Results written to `eval_comprehensive_results.json`.

---

### Run LLM-Guided Policy (PyBullet)

```bash
cd morpho_gnn_robot/Training_GNN

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
cd morpho_gnn_robot/Training_GNN

python test_morphology_transfer.py

python test_morphology_transfer.py ./checkpoints/gnn_ppo_final.pt
```

This loads the quadruped-trained weights, expands `log_std` from 12→18, remaps running norm statistics, and runs the hexapod in PyBullet GUI in real time.

---

### Generate Hexapod URDF

```bash
cd morpho_gnn_robot/Training_GNN
python generate_hexapod.py
```

---

### Inspect a Checkpoint

```bash
cd morpho_gnn_robot/Training_GNN
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

r_alive     =  0.1  (constant survival bonus)
r_vel       =  4.0 * exp(-2 * (vx - cmd_vx)²)   ← boosted from 1.5 to break crouching
             + 1.5 * exp(-2 * (wy - cmd_wy)²)
             - 0.5  (if cmd_vx > 0.1 and actual vx < 0.15 — standing-still penalty)
r_stability = -1.0 * (roll² + pitch²)
             - 0.5 * |lateral_vel|
r_torque    = -5e-6 * sum(torques²)
r_contact   = -0.5 if non-foot link touches ground (+ terminates)
fall        = -250.0 + episode terminates (base_height < 0.25 m or |roll/pitch| > 0.8 rad)
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
- `installers/` — OS setup scripts (distributed separately alongside `hetero_gnn_project.tar`)

---

## Deployment Engineering Notes

Several training/deployment mismatches were discovered and resolved during Gazebo integration:

| Issue | Root Cause | Fix |
|---|---|---|
| Robot drifts in circles | GNN trained with `wy=0` only → inherent gait asymmetry (back legs produce ~1.5× more action than front legs) | Yaw-rate PI controller via post-action HAA joint offset |
| Front legs barely visible | Training used PD gains `Kp=150, Kd=5`; Gazebo uses `Kp=1500, Kd=150` (10× stiffer) | Raised `MAX_CMD_STEP` from 0.2→0.5 and `ACTION_SMOOTH_ALPHA` from 0.8→0.6 |
| OOD `wy` injection | Previous fix injected `wy≠0` into GNN command; GNN was never trained with `wy≠0` | Clamp `wy` to `±0.3` at GNN input; yaw correction applied only to HAA joints post-action |
| Floor detected as obstacle | Depth ROI included lower 55% of image where floor appears; `front` pixel unfiltered | Tighten ROI to top 40%; filter front pixel by `MIN_OBSTACLE_DIST=2.0m` |
| LLM never called | `path_clear` check required `closest > 3.0m` but floor always at ~2m → LLM throttled | LLM still called for navigation; reactive fallback handles obstacle-clearing |

---

## Research Status

| Component | Status |
|---|---|
| MLP baseline training (5 seeds × 12M steps) | ✅ Complete — `mlp_ppo_final.pt` |
| MLP transfer failure demo | ✅ Complete |
| GNN policy training (5 seeds × 12M steps) | ✅ Complete — `gnn_ppo_final.pt` |
| Multi-seed stability (4/5 seeds stable) | ✅ Documented — Seed 1 entropy collapse at 8.97M steps |
| Zero-shot hexapod transfer (PyBullet) | ✅ 106 ± 25 reward, 47 steps survival |
| Unitree Aliengo/Go1 architectural compatibility | ✅ GNN executes (MLP cannot load); perf bounded by URDF fidelity |
| Terrain robustness evaluation | ✅ 95% success at 5° slope (zero-shot) |
| Hexapod 500K fine-tuning (Kaggle CPU) | ✅ 3.8× reward gain (110 → 416), survival 47 → 193 steps |
| Hexapod URDF generator | ✅ Complete |
| LLM planning layer (standalone + ROS2) | ✅ Complete — Qwen 2.5 7B via Ollama |
| ROS2 / Gazebo: MLP policy deployment | ✅ Stable locomotion + obstacle avoidance |
| ROS2 / Gazebo: GNN policy deployment | ✅ Forward trot + LLM-guided hexapod navigation |
| Yaw-rate PI correction (HAA joint offset) | ✅ Eliminates circular drift |
| PyBullet stderr suppressor | ✅ fd-level redirect — clean training logs |
| Publication plots | ✅ Generated |
| **Research paper** | ✅ Written |

---

## License

See [LICENSE](./LICENSE).

---

*Research project by Chaitanya Parate — MIT World Peace University, Pune.*
*Conducted as part of an independent research initiative in robot learning and morphological generalization.*

