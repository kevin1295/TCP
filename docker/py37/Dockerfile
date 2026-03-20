# 基础镜像：Ubuntu 20.04 + Conda + CUDA 11.3（匹配pytorch 1.10.1）
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 避免交互性输入
ENV DEBIAN_FRONTEND=noninteractive
# 设置Conda环境变量
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
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
    libjpeg-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 安装Miniconda（适配Python 3.7）
RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    # 初始化Conda
    conda init bash && \
    # 清理缓存
    conda clean -afy

# 4. 配置Conda清华源（核心：加速Conda包下载）
COPY tuna-mirror/.condarc /root/.condarc
RUN mkdir -p /root/.config/pip/
COPY tuna-mirror/pip.config /root/.config/pip/pip.config

# 5. 复制environment.yml并创建Conda环境
COPY environment.yml /tmp/environment.yml
# 创建环境时强制使用清华源，同时配置pip源
RUN conda env create -f /tmp/environment.yml -n TCP && \
    conda clean -afy && \
    rm /tmp/environment.yml && \
    # 激活环境（永久生效）
    /bin/bash -c "source activate TCP"

# 6. 安装CARLA Python API
COPY carla*.egg /tmp/
RUN bash -c "source activate TCP && \
    easy_install /tmp/carla*.egg && \
    rm /tmp/carla*.egg"

# 7. 创建工作目录（后续通过挂载映射项目代码）
WORKDIR /workspace/TCP
RUN echo "conda activate TCP" >> ~/.bashrc

# 8. 配置CARLA/ScenarioRunner/Leaderboard环境变量
ENV CARLA_ROOT=/workspace/carla
ENV SCENARIO_RUNNER_ROOT=/workspace/TCP/scenario_runner
ENV LEADERBOARD_ROOT=/workspace/TCP/leaderboard
ENV TEAM_CODE_ROOT=/workspace/TCP/team_code
ENV PYTHONPATH=/workspace/TCP:\
$PYTHONPATH:\
$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10.1-py3.7-linux-x86_64.egg:\
$CARLA_ROOT/PythonAPI/carla/agents:\
$CARLA_ROOT/PythonAPI/carla:\
$CARLA_ROOT/PythonAPI:\
$SCENARIO_RUNNER_ROOT:\
$LEADERBOARD_ROOT

# 9. 设置默认启动命令
CMD ["/bin/bash"]