export TORCH=2.4.1
export TORCHVISION=0.19.1
export CUDA=cu121
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple torch==${TORCH} torchvision==${TORCHVISION}
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch_geometric
pip install spconv-cu121
