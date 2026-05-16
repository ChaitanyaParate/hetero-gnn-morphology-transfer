# 🍎 macOS Setup — Hetero-GNN Morphology Transfer

This folder contains the setup application for **macOS 12 Monterey and later**.

> ⚠️ **Performance Warning:** This Docker image is compiled for `x86_64` and requires an NVIDIA GPU for full performance.
> - **Apple Silicon (M1/M2/M3/M4):** Runs via Rosetta 2 emulation — very slow, PyTorch falls back to CPU.
> - **Intel Mac:** PyTorch falls back to CPU. Gazebo may have limited rendering quality.
>
> macOS is supported for inspection and testing only. For full simulation performance, use a Linux machine with an NVIDIA GPU.

---

## 📋 Requirements

| Requirement | Details |
|---|---|
| **OS** | macOS 12 Monterey or later |
| **Docker** | Docker Desktop for Mac |
| **Display** | XQuartz (for Gazebo GUI) |
| **RAM** | 16 GB minimum |
| **Disk** | ~10 GB free for Docker image |

---

## ⬇️ Step 1 — Download the Docker Image

The pre-built image (`hetero_gnn_project.tar`, **8.3 GB**) is archived on Zenodo:

**Primary — Zenodo (permanent DOI):**
```bash
wget "https://zenodo.org/records/20187567/files/hetero_gnn_project.tar"
```
https://doi.org/10.5281/zenodo.20187567

**Alternative — Google Drive:**
```bash
pip install gdown && gdown 1tI2VpsGHoFGOhWkKnZxwH05tAVnjoXd5 -O hetero_gnn_project.tar
```

---

## 📁 Step 2 — Place in This Folder

After downloading, copy the archive here:

```
installers/macOS/
├── setup_macOS.app         ← double-click this in Finder  ✅
├── README.md               ← you are reading this
└── hetero_gnn_project.tar  ← place it here  ✅
```

---

## 🚀 How to Run

### Step 1 — Install Prerequisites (one-time setup)

**1a. Install XQuartz** (required for Gazebo GUI):
- Download from https://www.xquartz.org/
- Install and **reboot your Mac**
- Open XQuartz → **Preferences → Security** → ✅ check **"Allow connections from network clients"**

**1b. Install Docker Desktop**:
- Download from https://www.docker.com/products/docker-desktop/
- Install, open it, and wait for the whale icon to appear in the menu bar ✅

### Step 2 — Run the Setup App

**Double-click `setup_macOS.app` in Finder.**

> If macOS shows *"cannot be opened because it is from an unidentified developer"*:
> - Right-click the app → **"Open"** → click **"Open"** in the dialog.
> - This only needs to be done once.

A Terminal window will open automatically and walk you through:
1. ✅ Verifying XQuartz and Docker Desktop are installed
2. ✅ Loading the Docker image from `hetero_gnn_project.tar`
3. ✅ Launching the simulation container with software-rendered Gazebo

---

## 💻 Inside the Container

Once the container starts, you will be in the ROS2 workspace:

```
root@...:/workspace/morpho_gnn_robot/morpho_ros2_ws#
```

From here you can run:

```bash
# Launch Gazebo (software-rendered on macOS)
ros2 launch morpho_robot morpho_robot.launch.py

# Run zero-shot evaluation
python3 /workspace/morpho_gnn_robot/Training_GNN/eval_comprehensive.py
```

Type `exit` to stop and remove the container when finished.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| App shows "unidentified developer" | Right-click app → Open → Open |
| XQuartz window doesn't open | Reboot after XQuartz install; re-check the "Allow connections" setting |
| Gazebo crashes with Segfault | Inside the container, run `export GZ_GUI_RENDER_ENGINE=ogre` before launching |
| `docker: command not found` | Docker Desktop is not running — open it from Applications and wait for it to start |
| Very slow performance | Expected on Apple Silicon (Rosetta emulation). Try running on a Linux machine for best results |
