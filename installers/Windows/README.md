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
installers/Windows/
├── setup_windows.bat       ← double-click this to run  ✅
├── setup_windows.ps1       ← the main script (runs automatically)
├── README.md               ← you are reading this
└── hetero_gnn_project.tar  ← place it here  ✅
```

---

## 🚀 How to Run

### Step 1 — Install WSL2 + Ubuntu

> ⚠️ **Important:** Having WSL2 enabled is not enough — you also need an actual Linux distribution installed and set as default. Many Windows machines have the WSL framework but no distro, which causes the script to fail.

Open **PowerShell as Administrator** and run all three commands:

```powershell
# 1. Install Ubuntu (installs the Linux distro, not just the framework)
wsl --install -d Ubuntu

# 2. Set Ubuntu as the default WSL distro
wsl --set-default Ubuntu

# 3. Verify Ubuntu is listed and marked as default (*)
wsl --list --verbose
```

Restart your PC when prompted after step 1.

---

### Step 2 — Install Docker Desktop + Enable WSL Integration

1. Download and install from: https://www.docker.com/products/docker-desktop/
2. Open Docker Desktop and wait for the engine to start (whale icon in the taskbar).
3. Go to **Settings → Resources → WSL Integration**.
4. Click **Refresh** to update the distro list (Ubuntu should now appear).
5. Toggle **ON** the switch next to **Ubuntu**.
6. Click **Apply & Restart**.

> ⚠️ If Ubuntu doesn't appear in the list, click Refresh — Docker Desktop caches the distro list and may not detect newly installed distros automatically.

---

### Step 3 — Run the Setup

**Double-click `setup_windows.bat`**

> If Windows shows a SmartScreen warning, click **"More info"** → **"Run anyway"**. The script will request Administrator privileges automatically.

The script will automatically:
1. ✅ Verify WSL2 + Ubuntu are installed
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
python3 /workspace/morpho_gnn_robot/Training_GNN/eval_comprehensive.py
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
