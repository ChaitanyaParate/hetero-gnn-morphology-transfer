# 📦 Installers — Hetero-GNN Morphology Transfer

Choose the folder matching your operating system. Each folder is self-contained and includes a detailed `README.md` with step-by-step instructions.

```
installers/
├── 🐧 Linux/       → Ubuntu / Debian Linux with NVIDIA GPU  (recommended)
├── 🪟 Windows/     → Windows 10/11 via Docker Desktop + WSL2
└── 🍎 macOS/       → macOS 12+ via Docker Desktop + XQuartz (CPU fallback only)
```

## Quick Start

**Step 1 — Download the Docker image (8.3 GB)**

**[⬇ Download hetero_gnn_project.tar](https://drive.google.com/file/d/1tI2VpsGHoFGOhWkKnZxwH05tAVnjoXd5/view?usp=sharing)**

Or via terminal:
```bash
pip install gdown
gdown 1tI2VpsGHoFGOhWkKnZxwH05tAVnjoXd5 -O hetero_gnn_project.tar
```

**Step 2 — Place the `.tar` into the folder matching your OS, then run the setup script.**

| OS | File to run | How to run |
|---|---|---|
| Linux | `Linux/setup_linux.sh` | `chmod +x setup_linux.sh && ./setup_linux.sh` |
| Windows | `Windows/setup_windows.bat` | Double-click |
| macOS | `macOS/setup_macOS.app` | Double-click in Finder |

## Hardware Requirements

| Platform | GPU | Expected Performance |
|---|---|---|
| Linux (Ubuntu) | NVIDIA GPU (required) | ✅ Full CUDA + Gazebo hardware rendering |
| Windows 10/11 | NVIDIA GPU (required) | ✅ Full CUDA via WSL2 GPU passthrough |
| macOS | None (not supported) | ⚠️ CPU fallback only — limited performance |
