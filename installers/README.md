# 📦 Installers — Hetero-GNN Morphology Transfer

Choose the folder matching your operating system. Each folder is self-contained and includes a detailed `README.md` with step-by-step instructions.

```
installers/
├── 🐧 Linux/       → Ubuntu / Debian Linux with NVIDIA GPU  (recommended)
├── 🪟 Windows/     → Windows 10/11 via Docker Desktop + WSL2
└── 🍎 macOS/       → macOS 12+ via Docker Desktop + XQuartz (CPU fallback only)
```

## Quick Start

1. **Copy** `hetero_gnn_project.tar` into the folder matching your OS.
2. **Run** the setup script inside that folder.
3. **Read** the `README.md` in that folder if you run into any issues.

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
