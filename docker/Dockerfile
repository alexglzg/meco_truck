# Base image
FROM osrf/ros:foxy-desktop

# Prevent interactive prompts during installation [cite: 1]
ENV DEBIAN_FRONTEND=noninteractive

# 1. Update the GPG key and install ALL apt dependencies in a single step.
# Combining these saves disk space and stops 'apt-get update' from running 4 separate times.
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys AD19BAB3CBF125EA
RUN apt-get update || true && apt-get install -y curl gnupg2 && \
    curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA' | apt-key add - && \
    apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ros-foxy-rviz2 \
    ros-foxy-tf2-tools \
    ros-foxy-nav2-msgs \
    ros-foxy-pcl-conversions \
    ros-foxy-pcl-ros \
    ros-foxy-navigation2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Initialize and update rosdep 
RUN rosdep init || true && \
    rosdep update

# 3. Install all Python packages via pip [cite: 2, 3]
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    transforms3d \
    casadi \
    rockit-meco \
    pybullet

# 4. Set up the ROS workspace 
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# 5. Automatically source ROS environments when opening a terminal 
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# Default command to run when starting the container 
CMD ["bash"]