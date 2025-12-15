# Dockerfile for CropClassifier ML Environment
# Python 3.10.18, CUDA 12.2, TensorFlow 2.17.0, PySpark, JupyterLab

# Base image with CUDA 12.3 and Ubuntu 22.04
# TensorFlow 2.17.0 requires CUDA 12.3, not 12.2
# Using base image and installing cuDNN separately for better compatibility
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Metadata
LABEL maintainer="federico"
LABEL description="CropClassifier ML environment with GPU support, PySpark, and JupyterLab"
LABEL cuda.version="12.3"
LABEL python.version="3.10.18"
LABEL tensorflow.version="2.17.0"

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Argentina/Buenos_Aires

# Update system and install base dependencies
RUN apt-get update && apt-get install -y \
    # Python
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    # Development tools
    git \
    wget \
    curl \
    vim \
    nano \
    build-essential \
    cmake \
    # Geospatial libraries
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    libproj-dev \
    libgeos-dev \
    # Java for PySpark
    openjdk-11-jdk \
    # System utilities
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Configure Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Configure environment variables for CUDA and TensorFlow
ENV CUDA_HOME=/usr/local/cuda-12.3
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_THREAD_MODE=gpu_private
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Configure JAVA_HOME for Spark/PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=${JAVA_HOME}/bin:${PATH}

# Create workspace directory
WORKDIR /workspace

# Copy requirements.txt
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt

# Install JupyterLab and useful extensions
RUN pip3 install --no-cache-dir \
    jupyterlab==4.0.9 \
    jupyter-resource-usage \
    jupyterlab-git \
    ipywidgets \
    notebook \
    nbconvert

# Create directories for project structure
RUN mkdir -p /workspace/notebooks \
             /workspace/data \
             /workspace/models \
             /workspace/results \
             /workspace/preprocessing \
             /workspace/config

# Configure JupyterLab (running as root)
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.notebook_dir = '/workspace'" >> ~/.jupyter/jupyter_lab_config.py

# Expose JupyterLab port
EXPOSE 8888

# Set working directory
WORKDIR /workspace

# Default command: start bash (allows running scripts)
# To start JupyterLab instead, override with docker-compose
CMD ["/bin/bash"]
