# 基础镜像：Ubuntu 20.04 + Conda + CUDA 11.3（匹配pytorch 1.10.1）
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# 避免交互性输入
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 1. 替换apt源为清华源（加速系统依赖安装）
RUN ls /etc/apt/
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak
COPY tuna-mirror/sources.list /etc/apt/sources.list

# 2. 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    build-essential \
    openssh-client \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. 配置Conda清华源（核心：加速Conda包下载）
RUN conda config --set show_channel_urls yes
COPY tuna-mirror/.condarc /root/.condarc
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
COPY tuna-mirror/pip.config /root/.config/pip/pip.config

COPY torch-staff.sh /tmp/torch-staff.sh
COPY requirements.txt /tmp/requirements.txt
COPY flash_attn*.whl /tmp/
RUN bash /tmp/torch-staff.sh
RUN pip install /tmp/flash_attn*.whl
RUN pip install -r /tmp/requirements.txt

# 7. 创建工作目录（后续通过挂载映射项目代码）
WORKDIR /workspace/TCP

# 9. 设置默认启动命令
CMD ["/bin/bash"]