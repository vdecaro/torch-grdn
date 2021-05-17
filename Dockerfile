FROM nvidia/cuda:11.3.0-devel-ubuntu18.04

# All users can use /home/user as their home directory.
RUN mkdir /home/user

ENV HOME=/home/user
RUN chmod 777 /home/user

RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git

# Install Miniconda.
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.7 environment.
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py37 python=3.7 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Install JupyterLab.
RUN conda install -c conda-forge jupyterlab

ENV TORCH=1.8.0
ENV CUDA=cu111

RUN pip install -U "ray[default]" \
 && pip install -U "ray[tune]" \
 && pip install -U nvgpu \
 && pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html\
 && pip install -U scikit-learn

RUN pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
 && pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
 && pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
 && pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html \
 && pip install torch-geometric


CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

