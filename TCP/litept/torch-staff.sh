export TORCH=2.9.1
export TORCHVISION=0.24.1
export CUDA=cu129
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple torch==${TORCH} torchvision==${TORCHVISION}
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch_geometric
pip install spconv-${CUDA}
