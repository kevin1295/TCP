# 基础镜像：Ubuntu 22.04 + CUDA 12.8（匹配pytorch 2.9.0）
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# 避免交互性输入
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 1. 替换apt源为清华源（加速系统依赖安装）
RUN ls /etc/apt/
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
COPY tuna-mirror/sources.list /etc/apt/sources.list

# 2. 安装系统依赖（CARLA/ScenarioRunner运行必需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libomp-dev \
    openssh-client \
    libjpeg-dev
RUN apt-get install -y --no-install-recommends \
    python-is-python3 \
    python3-pip \
    python3-venv \
    python3-dev
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.config/pip/
COPY tuna-mirror/pip.config /root/.config/pip/pip.config
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 5. 复制environment.yml并创建Conda环境
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install torch-geometric && \
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
    
# 6. 安装CARLA Python API
COPY carla*.whl /tmp/
RUN pip install /tmp/carla*.whl && \
    rm /tmp/carla*.whl

# 7. 创建工作目录（后续通过挂载映射项目代码）
WORKDIR /workspace/TCP

# 8. 配置CARLA/ScenarioRunner/Leaderboard环境变量
ENV CARLA_ROOT=/workspace/carla
ENV SCENARIO_RUNNER_ROOT=/workspace/TCP/scenario_runner
ENV LEADERBOARD_ROOT=/workspace/TCP/leaderboard
ENV TEAM_CODE_ROOT=/workspace/TCP/team_code
ENV PYTHONPATH=/workspace/TCP:\
$PYTHONPATH:\
$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.whl:\
$CARLA_ROOT/PythonAPI/carla/agents:\
$CARLA_ROOT/PythonAPI/carla:\
$CARLA_ROOT/PythonAPI:\
$SCENARIO_RUNNER_ROOT:\
$LEADERBOARD_ROOT

# 9. 设置默认启动命令
CMD ["/bin/bash"]