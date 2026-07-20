FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
COPY sources.list /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    build-essential \
    openssh-client \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python-is-python3 \
    ros-humble-derived-object-msgs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
COPY pip.config /root/.config/pip/pip.config
COPY requirements.txt /tmp/requirements.txt
COPY carla*.whl /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install openvino==2026.0.0
RUN pip3 install /tmp/carla*.whl
RUN rm /tmp/requirements.txt

COPY ros-bridge /opt/ros-bridge
RUN cd /opt/ros-bridge && pip3 install -r requirements.txt && bash install_dependencies.sh
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash \
    && cd /opt/ros-bridge \
    && colcon build --symlink-install"

RUN echo "# Load ROS Humble environment" >> ~/.bashrc \
    && echo "set +e" >> ~/.bashrc \
    && echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc \
    && echo "source /opt/ros-bridge/install/setup.bash" >> ~/.bashrc \
    && echo "source /root/workspace/ros_ws/install/setup.bash" >> ~/.bashrc
    
WORKDIR /root/workspace
