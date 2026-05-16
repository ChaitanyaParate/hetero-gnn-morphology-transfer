#!/usr/bin/env bash
# ==============================================================================
# Setup Script: Hetero-GNN Morphology Transfer (Ubuntu / Linux)
# Place hetero_gnn_project.tar in the same folder as this script, then run:
#   chmod +x setup_linux.sh && ./setup_linux.sh
# ==============================================================================

set -e

# --- Colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${CYAN}
╔══════════════════════════════════════════════════════════════╗
║    Hetero-GNN Morphology Transfer — Ubuntu/Linux Setup       ║
║    ROS2 Jazzy  ·  PyTorch CUDA  ·  Gazebo Harmonic           ║
╚══════════════════════════════════════════════════════════════╝
${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="relational_bias_for_morphological_generalization-hetero-gnn:latest"
TAR_FILE="hetero_gnn_project.tar"

# ------------------------------------------------------------------------------
# Step 1: Verify NVIDIA Drivers
# ------------------------------------------------------------------------------
echo -e "${BOLD}[Step 1/5]${NC} Verifying NVIDIA GPU drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗  nvidia-smi not found. Installing NVIDIA drivers...${NC}"
    sudo ubuntu-drivers autoinstall
    echo -e "${YELLOW}⚠  Drivers installed. Please REBOOT and re-run this script.${NC}"
    exit 1
fi
echo -e "${GREEN}✓  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)${NC}"

# ------------------------------------------------------------------------------
# Step 2: Install Docker Engine
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 2/5]${NC} Checking Docker Engine..."
if ! command -v docker &> /dev/null; then
    echo -e "${CYAN}  Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh && rm /tmp/get-docker.sh
    echo -e "${GREEN}✓  Docker installed.${NC}"
else
    echo -e "${GREEN}✓  Docker already installed: $(docker --version)${NC}"
fi

# ------------------------------------------------------------------------------
# Step 3: Install & Configure NVIDIA Container Toolkit
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 3/5]${NC} Checking NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo -e "${CYAN}  Adding NVIDIA package repos...${NC}"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo -e "${GREEN}✓  NVIDIA Container Toolkit installed and configured.${NC}"
else
    echo -e "${GREEN}✓  NVIDIA Container Toolkit already installed.${NC}"
fi

# ------------------------------------------------------------------------------
# Step 4: Load Docker Image
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 4/5]${NC} Loading Docker image..."

if [ ! -f "${SCRIPT_DIR}/${TAR_FILE}" ]; then
    echo -e "${RED}✗  ERROR: '${TAR_FILE}' not found in:${NC}"
    echo -e "       ${SCRIPT_DIR}"
    echo -e ""
    echo -e "${YELLOW}  Download it from Google Drive (8.3 GB):${NC}"
    echo -e "  https://drive.google.com/file/d/1tI2VpsGHoFGOhWkKnZxwH05tAVnjoXd5/view?usp=sharing"
    echo -e ""
    echo -e "${CYAN}  Or via terminal:${NC}"
    echo -e "    pip install gdown && gdown 1tI2VpsGHoFGOhWkKnZxwH05tAVnjoXd5 -O ${SCRIPT_DIR}/${TAR_FILE}"
    echo -e ""
    echo -e "${YELLOW}  Then re-run this script.${NC}"
    exit 1
fi

# Check if image is already loaded to avoid re-loading
if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMAGE_NAME}"; then
    echo -e "${GREEN}✓  Image already loaded: ${IMAGE_NAME}${NC}"
else
    echo -e "${CYAN}  Loading ~8.3 GB image — this may take a few minutes...${NC}"
    sudo docker load -i "${SCRIPT_DIR}/${TAR_FILE}"
    echo -e "${GREEN}✓  Docker image loaded.${NC}"
fi

# ------------------------------------------------------------------------------
# Step 5: Launch the Container
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}[Step 5/5]${NC} Launching simulation container..."
echo -e "${CYAN}  Granting X11 display access...${NC}"
xhost +local:root

echo -e "${YELLOW}
  ┌─────────────────────────────────────────────────────────┐
  │  Container is starting. You will be dropped into the    │
  │  ROS2 workspace at:                                     │
  │    /workspace/morpho_gnn_robot/morpho_ros2_ws           │
  │                                                         │
  │  Type 'exit' when finished to remove the container.     │
  └─────────────────────────────────────────────────────────┘
${NC}"

sudo docker run -it --rm \
    --network host \
    --gpus all \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e QT_QPA_PLATFORM=xcb \
    -e __NV_PRIME_RENDER_OFFLOAD=1 \
    -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev/dri:/dev/dri:rw \
    "${IMAGE_NAME}"

echo -e "\n${GREEN}✓  Container session ended. Environment cleaned up.${NC}"
