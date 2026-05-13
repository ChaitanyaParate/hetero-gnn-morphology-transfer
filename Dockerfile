# Start with the official ROS2 Jazzy image
FROM osrf/ros:jazzy-desktop

# Set up the working directory
WORKDIR /workspace

# Install system dependencies needed for building and running
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies natively (Added a 1000-second timeout for massive PyTorch wheels)
RUN pip3 install --default-timeout=1000 --break-system-packages --no-cache-dir \
    torch torchvision \
    torch-geometric \
    pybullet gymnasium scipy numpy wandb \
    ollama
RUN apt-get update && apt-get install -y ros-jazzy-ros-gz

# Source ROS2 and build the colcon workspace 
COPY . /workspace
RUN cd morpho_gnn_robot
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    cd /workspace/morpho_gnn_robot && \
    rm -rf build/ install/ log/ && \
    colcon build --symlink-install --packages-select morpho_robot"

# Add the ROS2 setup scripts to bashrc so they load automatically
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
RUN echo "source /workspace/morpho_gnn_robot/install/setup.bash" >> ~/.bashrc

# Default command when the container starts
CMD ["/bin/bash"]