# ==============================================================================
# Setup Script: Hetero-GNN Morphology Transfer (Windows 10/11)
# Double-click setup_windows.bat to run (it calls this script automatically).
# ==============================================================================

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step { param($n, $msg) Write-Host "`n[Step $n]" -ForegroundColor Cyan -NoNewline; Write-Host " $msg" -ForegroundColor White }
function Write-Ok   { param($msg) Write-Host "  [OK]  $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  [!!]  $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "  [ERR] $msg" -ForegroundColor Red; Read-Host "Press Enter to exit"; exit 1 }
function Write-Info { param($msg) Write-Host "  -->   $msg" -ForegroundColor Gray }

Write-Host @"
+--------------------------------------------------------------+
|   Hetero-GNN Morphology Transfer -- Windows 10/11 Setup     |
|   ROS2 Jazzy  .  PyTorch CUDA  .  Gazebo Harmonic           |
|   (via Docker Desktop + WSL2)                                |
+--------------------------------------------------------------+
"@ -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# Step 1: Verify WSL2 and Docker Desktop
# ------------------------------------------------------------------------------
Write-Step "1/3" "Verifying WSL2 and Docker Desktop..."

$wslCheck = wsl --status 2>&1
if ($LASTEXITCODE -ne 0 -or ($wslCheck -join '') -match "not installed") {
    Write-Warn "WSL2 is not installed. Attempting to install now..."
    Write-Info "A reboot is required. Please re-run this script after rebooting."
    Start-Process powershell -Verb RunAs -ArgumentList "-Command wsl --install" -Wait
    Write-Warn "Reboot complete? Re-run setup_windows.bat to continue."
    Read-Host "Press Enter to exit"
    exit 0
}
Write-Ok "WSL2 is installed."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Warn "Docker Desktop not found. Opening download page..."
    Start-Process "https://www.docker.com/products/docker-desktop/"
    Write-Warn "Install Docker Desktop, enable WSL2 integration, then re-run this script."
    Read-Host "Press Enter to exit"
    exit 0
}
Write-Ok "Docker Desktop detected: $(docker --version)"

# ------------------------------------------------------------------------------
# Step 2: Load the Docker Image
# ------------------------------------------------------------------------------
Write-Step "2/3" "Loading Docker image..."

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$TarFile   = "hetero_gnn_project.tar"
$TarPath   = Join-Path $ScriptDir $TarFile
$ImageName = "relational_bias_for_morphological_generalization-hetero-gnn:latest"

if (-not (Test-Path $TarPath)) {
    Write-Err "ERROR: '$TarFile' not found in:`n       $ScriptDir`n`n  Copy 'hetero_gnn_project.tar' into this folder and re-run."
}

# Convert Windows paths to WSL paths (C:\Users\... -> /mnt/c/Users/...)
$WslTarPath   = ($TarPath   -replace '\\', '/') -replace '^([A-Za-z]):', { "/mnt/$($_.Value.ToLower().TrimEnd(':'))" }
$WslScriptDir = ($ScriptDir -replace '\\', '/') -replace '^([A-Za-z]):', { "/mnt/$($_.Value.ToLower().TrimEnd(':'))" }

# Skip load if image already exists
$existingImage = wsl -- docker images --format "{{.Repository}}:{{.Tag}}" 2>$null | Where-Object { $_ -eq $ImageName }
if ($existingImage) {
    Write-Ok "Image already loaded: $ImageName"
} else {
    Write-Info "Loading from: $WslTarPath"
    Write-Info "This may take a few minutes for the ~8.1 GB image..."
    wsl -- docker load -i "$WslTarPath"
    if ($LASTEXITCODE -ne 0) { Write-Err "docker load failed. Check the output above." }
    Write-Ok "Docker image loaded successfully."
}

# ------------------------------------------------------------------------------
# Step 3: Launch the Container
# ------------------------------------------------------------------------------
Write-Step "3/3" "Launching simulation container..."
Write-Host @"

  ┌─────────────────────────────────────────────────────────┐
  │  Container is starting. You will be dropped into the    │
  │  ROS2 workspace at:                                     │
  │    /workspace/morpho_gnn_robot/morpho_ros2_ws           │
  │                                                         │
  │  Type 'exit' when finished to remove the container.     │
  └─────────────────────────────────────────────────────────┘

"@ -ForegroundColor Yellow

$DockerCmd = (
    "docker run -it --rm",
    "--network host",
    "--gpus all",
    "-e DISPLAY=`$DISPLAY",
    "-e QT_X11_NO_MITSHM=1",
    "-e NVIDIA_DRIVER_CAPABILITIES=all",
    "-v /tmp/.X11-unix:/tmp/.X11-unix:rw",
    $ImageName
) -join " "

wsl -- bash -c $DockerCmd

Write-Host "`n  [OK] Container session ended. Environment cleaned up." -ForegroundColor Green
Read-Host "`nPress Enter to close this window"
