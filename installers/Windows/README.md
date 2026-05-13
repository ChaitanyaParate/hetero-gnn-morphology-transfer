# 🪟 Windows Setup — Hetero-GNN Morphology Transfer

This folder contains the automated setup scripts for **Windows 10 (Build 19044+) and Windows 11**, using Docker Desktop with WSL2.

---

## 📋 Requirements

| Requirement | Details |
|---|---|
| **OS** | Windows 11 *(recommended)* or Windows 10 Build 19044+ |
| **GPU** | NVIDIA GPU with latest Game Ready or Studio drivers |
| **RAM** | 16 GB minimum, 32 GB recommended |
| **Disk** | ~10 GB free for Docker image |
| **WSL** | WSL2 with an Ubuntu distribution installed |

> **Windows 11 is strongly recommended.** It includes native WSLg support for GUI apps (Gazebo), so no extra X-Server software is needed.

---

## 📁 What to Place in This Folder

Before running the script, copy the Docker image archive here:

```
installers/Windows/
├── setup_windows.bat       ← double-click this to run  ✅
├── setup_windows.ps1       ← the main script (runs automatically)
├── README.md               ← you are reading this
└── hetero_gnn_project.tar  ← copy the image archive here  ✅
```

---

## 🚀 How to Run

### Step 1 — Pre-check: Install WSL2 (if not already installed)
Open **PowerShell as Administrator** and run:
```powershell
wsl --install
```
Restart your PC when prompted. This installs Ubuntu via WSL2 automatically.

### Step 2 — Pre-check: Install Docker Desktop
Download and install from: https://www.docker.com/products/docker-desktop/

After installing, open Docker Desktop → **Settings → Resources → WSL Integration** → enable it for your Ubuntu distro.

### Step 3 — Run the Setup

**Double-click `setup_windows.bat`**

> If Windows shows a SmartScreen warning, click **"More info"** → **"Run anyway"**. The script will request Administrator privileges automatically.

The script will automatically:
1. ✅ Verify WSL2 is installed
2. ✅ Verify Docker Desktop is running
3. ✅ Load the Docker image from `hetero_gnn_project.tar`
4. ✅ Launch the simulation container with GPU and Gazebo display

---

## 💻 Inside the Container

Once the container starts (inside the WSL terminal window), you will be in the ROS2 workspace:

```
root@...:/workspace/morpho_gnn_robot/morpho_ros2_ws#
```

From here you can run:

```bash
# Launch Gazebo simulation
ros2 launch morpho_robot morpho_robot.launch.py

# Run zero-shot evaluation
python3 /workspace/morpho_gnn_robot/Training_Location/eval_comprehensive.py
```

Type `exit` to stop and remove the container when finished.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `wsl: command not found` | Open PowerShell as Admin and run `wsl --install`, then reboot |
| `docker: command not found` | Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and enable WSL2 integration |
| Gazebo GUI doesn't open on Windows 10 | Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) and run it with "Disable access control" checked, then set `$env:DISPLAY = "localhost:0"` in your WSL terminal before launching the container |
| Script blocked by Windows Defender | Right-click `setup_windows.bat` → Properties → check "Unblock" at the bottom |
| Container exits immediately | Check that your NVIDIA GPU drivers are up to date on the **Windows host** |

---

## 🔑 Converting to a True `.exe` (optional)

If you need a single `.exe` file, install `ps2exe` on any Windows machine:
```powershell
Install-Module -Name ps2exe -Scope CurrentUser
Invoke-ps2exe .\setup_windows.ps1 .\setup_windows.exe -noConsole -title "Hetero-GNN Setup"
```
