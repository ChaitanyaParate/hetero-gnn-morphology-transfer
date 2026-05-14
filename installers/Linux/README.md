# 🐧 Linux Setup — Hetero-GNN Morphology Transfer

This folder contains the automated setup script for **Ubuntu 22.04 / 24.04** (or any modern Debian-based Linux distro with an NVIDIA GPU).

---

## 📋 Requirements

| Requirement | Details |
|---|---|
| **OS** | Ubuntu 22.04 LTS or 24.04 LTS (recommended) |
| **GPU** | NVIDIA GPU (GeForce / RTX / Quadro) — **strictly required** |
| **RAM** | 16 GB minimum, 32 GB recommended |
| **Disk** | ~10 GB free for Docker image |
| **Display** | X11 display server (default on Ubuntu desktop) |

---

## 📁 What to Place in This Folder

Before running the script, copy the Docker image archive here:

```
installers/Linux/
├── setup_linux.sh          ← this script
├── README.md               ← you are reading this
└── hetero_gnn_project.tar  ← copy the image archive here  ✅
```

---

## 🚀 How to Run

### Option A — Double-click (if your file manager supports it)
Right-click `setup_linux.sh` → **"Run as Program"** (Ubuntu Files / Nautilus).

### Option B — Terminal (recommended)

```bash
# 1. Open a terminal in this folder
cd path/to/installers/Linux

# 2. Make the script executable (only needed once)
chmod +x setup_linux.sh

# 3. Run the setup
./setup_linux.sh
```

The script will automatically:
1. ✅ Check & install NVIDIA drivers (if missing)
2. ✅ Install Docker Engine (if missing)
3. ✅ Install & configure the NVIDIA Container Toolkit
4. ✅ Load the Docker image from `hetero_gnn_project.tar`
5. ✅ Launch the simulation container with GPU and Gazebo display

---

## 💻 Inside the Container

Once the container starts, you will be dropped into the ROS2 workspace:

```
root@...:/workspace/morpho_gnn_robot/morpho_ros2_ws#
```

From here you can launch the Gazebo simulation or run evaluation scripts:

```bash
# Launch Gazebo simulation
ros2 launch morpho_robot morpho_robot.launch.py

# Run zero-shot evaluation
python3 /workspace/morpho_gnn_robot/Training_GNN/eval_comprehensive.py
```

Type `exit` to stop and remove the container when finished.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `nvidia-smi: command not found` | Run `sudo ubuntu-drivers autoinstall` then reboot |
| `docker: command not found` | Re-run the script; it will install Docker automatically |
| Gazebo opens then immediately crashes | Ensure `nvidia-ctk runtime configure --runtime=docker` was run (script handles this) |
| `Cannot connect to X server` | Run `xhost +local:root` in your host terminal before launching |
| Container exits with `Permission denied` | Make sure you ran the script with `sudo` or that your user is in the `docker` group |
