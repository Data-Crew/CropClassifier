# 🛠️ Environment Setup Guide

This guide explains how to set up the CropClassifier development environment with GPU support for TensorFlow.

**Tested Configuration:**
- **OS**: Ubuntu 20.04 & 22.04
- **Python**: 3.10.18
- **NVIDIA Driver**: 565.57.01
- **CUDA**: 12.2
- **TensorFlow**: 2.17.0

---

## 📋 Choose Your Setup Method

### Option A: Virtual Environment (Local Development)
Recommended if you want to work directly on your machine with full control.

### Option B: Docker (Reproducibility & Portability)
Recommended for sharing environments or avoiding conflicts with system packages.

---

## ⚡ Quick Start

### Virtual Environment (5 minutes)

```bash
# 1. Create and activate environment
virtualenv venv --python=python3.10
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. Verify GPU
python config/gpu/check_gpu.py

# 4. Run your first command
bash cropclassifier.sh -action test -model simplecnn
```

### Docker (15 minutes first time, then instant)

```bash
# 1. Build image (only first time)
./docker-run.sh build

# 2. Start container
./docker-run.sh start

# 3. Verify GPU
./docker-run.sh exec 'python config/gpu/check_gpu.py'

# 4. Run your first command
./docker-run.sh exec 'bash cropclassifier.sh -action test -model simplecnn'

# 5. Or open a shell for interactive work
./docker-run.sh shell
```

---

## Option A: Virtual Environment Setup

### Prerequisites

- Python 3.10 installed
- NVIDIA drivers compatible with CUDA 12.2
- CUDA Toolkit 12.2 installed

### Step 1: Save Your Current NVIDIA Configuration

Before proceeding, verify that your NVIDIA/CUDA setup is compatible with TensorFlow 2.17.0. The project includes a script to help you manage and verify your configuration.

**Important**: Before creating your environment, save your current working NVIDIA/CUDA configuration:

```bash
# Save configuration (no changes to system)
./config/save_nvidia_config.sh -action save

# Or save AND freeze packages to prevent updates
./config/save_nvidia_config.sh -action freeze
```

**What this script does:**
- **`-action save`**: Creates a snapshot of your NVIDIA driver, CUDA, and TensorFlow versions
- **`-action freeze`**: Saves snapshot AND freezes all NVIDIA/CUDA packages with `apt-mark hold`
- **`-action unfreeze`**: Releases all frozen packages to allow updates

**Output**: Creates `environment_config_YYYYMMDD_HHMMSS.txt` with complete configuration details:
```
- CUDA Toolkit version
- NVIDIA driver version
- GPU information
- TensorFlow version
- List of installed NVIDIA/CUDA packages
- Current package freeze status

```

**Why freeze packages?**

TensorFlow is compiled for specific CUDA Toolkit versions. System updates can break your working setup by updating or removing CUDA/NVIDIA packages. Freezing prevents `apt` from updating these packages.

**Freeze packages** (prevent updates):

```bash
./config/save_nvidia_config.sh -action freeze
```

This will:
- Save your current configuration
- Use `apt-mark hold` to prevent all NVIDIA/CUDA packages from being updated
- Create a record of frozen packages in the configuration file

**Unfreeze packages** (allow updates):

```bash
./config/save_nvidia_config.sh -action unfreeze
```

This releases all NVIDIA/CUDA packages for updates. Use this when you want to update your CUDA/NVIDIA drivers.

**Check current status**:

```bash
# Just save current status without changing anything
./config/save_nvidia_config.sh -action save
```

The output will indicate whether packages are currently frozen or free to update.


### Step 2: Create Python 3.10 Virtual Environment

```bash
# Install virtualenv if needed
pip3 install virtualenv

# Create virtual environment with Python 3.10
virtualenv venv --python=python3.10

# Activate environment
source venv/bin/activate

# Verify Python version
python --version  
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install all dependencies
pip install -r requirements.txt
```

This installs:
- TensorFlow 2.17.0 with CUDA 12.2 support
- PySpark 3.5.5 for data processing
- Geospatial libraries (rasterio, geopandas, pyproj)
- All other project dependencies

### Step 4: Verify Installation

```bash
# Test TensorFlow GPU
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
TensorFlow: 2.17.0
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Option B: Docker Setup

This option provides a fully isolated and reproducible environment, ideal for deployment, collaboration, or avoiding conflicts with your host system.

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed (for GPU support)
- NVIDIA drivers compatible with CUDA 12.2

```bash
# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
# Logout and login to apply changes

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Build Docker Image

```bash
# Verify GPU support first
./docker-run.sh check

# Build image (takes ~10-15 minutes)
./docker-run.sh build
```

The image includes:
- Ubuntu 22.04
- CUDA 12.2
- Python 3.10
- TensorFlow 2.17.0 with GPU
- All dependencies from `requirements.txt`

### Run Bash Scripts in Docker

Start the main container for executing bash scripts:

```bash
# Start container in detached mode
docker compose up -d cropclassifier

# Or use the helper script
./docker-run.sh start
```

Connect to the container's shell:

```bash
docker compose exec cropclassifier /bin/bash

# Or use the helper script
./docker-run.sh shell
```

Once inside the container, you can run your scripts:

```bash
# Example: Download training data
bash build_training_data.sh multiple all

# Example: Train models
bash cropclassifier.sh -action train -model inception1d

Or execute commands directly:

```bash
# Execute single command without entering shell
./docker-run.sh exec 'bash build_training_data.sh unique 3'
./docker-run.sh exec 'bash cropclassifier.sh -action train -model inception1d'
```

### Run JupyterLab in Docker


Start JupyterLab for interactive development:

```bash
# Start JupyterLab container
docker compose --profile jupyter up -d jupyter

# Or use the helper script
./docker-run.sh jupyter
```

Access JupyterLab in your browser at: `http://localhost:8888`

View logs to see the access token (if needed):

```bash
docker compose logs jupyter
```

All notebooks in `./notebooks/` will be available with GPU support.


### Docker Helper Script Commands

The `docker-run.sh` script simplifies common Docker operations:

```bash
./docker-run.sh check      # Verify GPU support
./docker-run.sh build      # Build Docker image
./docker-run.sh start      # Start main container (bash mode)
./docker-run.sh jupyter    # Start JupyterLab
./docker-run.sh all        # Start both containers
./docker-run.sh shell      # Open shell in container
./docker-run.sh exec 'cmd' # Execute command in container (e.g. "bash build_training_data.sh multiple all")
./docker-run.sh logs       # Show container logs
./docker-run.sh stop       # Stop all containers
```

---

## 🧪 Verify GPU Support

The project includes diagnostic scripts to quickly verify GPU configuration.

### Quick Verification (Both Modes)

**Option 1: Full diagnostic script** (recommended)

```bash
# Virtual environment
python config/gpu/check_gpu.py

# Docker
./docker-run.sh exec 'python config/gpu/check_gpu.py'
```

This script checks:
- Environment variables (NVIDIA_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES)
- nvidia-smi availability and GPU detection
- TensorFlow GPU detection and configuration
- CUDA library paths
- Provides troubleshooting tips if GPU is not detected

**Option 2: Direct CUDA test** (for low-level debugging)

```bash
# Virtual environment
python config/gpu/test_cuda_direct.py

# Docker
./docker-run.sh exec 'python config/gpu/test_cuda_direct.py'
```

This script tests CUDA initialization directly via `libcuda.so`, useful when TensorFlow fails but nvidia-smi works. It helps diagnose `CUDA_ERROR_UNKNOWN` errors.

**Option 3: Quick Python test**

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("CUDA built:", tf.test.is_built_with_cuda())

# Quick GPU computation test
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("GPU test successful!")
    print(c.numpy())
```

### Expected Output

When GPU is working correctly:

```
============================================================
GPU DIAGNOSTIC CHECK
============================================================

📋 Environment Variables:
  NVIDIA_VISIBLE_DEVICES: all
  CUDA_VISIBLE_DEVICES: NOT SET
  LD_LIBRARY_PATH: /usr/local/cuda/lib64...

🔍 Checking nvidia-smi...
✅ nvidia-smi works!
  NVIDIA GeForce RTX 4060 Ti

🔍 Checking TensorFlow...
✅ TensorFlow version: 2.17.0
  CUDA built: True
  Physical GPUs found: 1
    GPU 0: /physical_device:GPU:0
  Logical GPUs: 1
✅ GPU memory growth configured successfully
```

---

## 🔧 Troubleshooting

### GPU Diagnostic Scripts

The project includes three diagnostic scripts in `config/gpu/`:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `check_gpu.py` | Full diagnostic | First check when GPU issues occur |
| `test_cuda_direct.py` | Low-level CUDA test | When TensorFlow fails but nvidia-smi works |
| `check_cuda_init_error.sh` | Fix Docker GPU errors | CUDA_ERROR_UNKNOWN in Docker |

### GPU Not Detected

**Step 1: Run full diagnostic**
```bash
python config/gpu/check_gpu.py
```

**Step 2: Check NVIDIA driver**
```bash
nvidia-smi
```

**Step 3: Check CUDA version**
```bash
nvcc --version  # Should show CUDA 12.2
```

**Step 4: Reinstall TensorFlow with CUDA** (if needed)
```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.17.0
```

### System Wants to Update NVIDIA Drivers

**Problem**: Ubuntu upgrade wants to change driver 565 → 575

**Solution**: Freeze your current working configuration

```bash
# Freeze all NVIDIA/CUDA packages
./config/save_nvidia_config.sh -action freeze
```

**To unfreeze later** (when TensorFlow supports new CUDA):
```bash
./config/save_nvidia_config.sh -action unfreeze
```

### Docker GPU Not Working

**Step 1: Run Docker-specific diagnostic**
```bash
sudo bash config/gpu/check_cuda_init_error.sh
```

This script:
- Checks NVIDIA driver status
- Verifies nvidia-container-toolkit installation
- Identifies processes blocking the GPU
- Restarts Docker service
- Tests basic GPU access

**Step 2: Manual verification**
```bash
# Verify nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Restart Docker
sudo systemctl restart docker
```

**Step 3: Restart container**
```bash
docker compose restart cropclassifier
./docker-run.sh exec 'python config/gpu/check_gpu.py'
```

### CUDA_ERROR_UNKNOWN in Docker

This error typically occurs after container crashes or host system events.

**Quick fix:**
```bash
# Run the diagnostic and fix script
sudo bash config/gpu/check_cuda_init_error.sh

# Restart container
docker compose restart cropclassifier
```

**If the problem persists:**
```bash
# Test CUDA directly
./docker-run.sh exec 'python config/gpu/test_cuda_direct.py'

# If test fails, reboot the host system
sudo reboot
```

### TensorFlow Detects 0 GPUs but nvidia-smi Works

This is usually a CUDA initialization issue. Run the direct CUDA test:

```bash
python config/gpu/test_cuda_direct.py
```

If `cuInit` fails with error code 999 (CUDA_ERROR_UNKNOWN):
1. Check if other processes are using the GPU
2. Restart Docker service (if using Docker)
3. Restart the host system as a last resort

### PySpark Out of Memory

Edit `preprocessing/spark_session.py`:

```python
.config("spark.driver.memory", "32g")     # Increase from 16g
.config("spark.executor.memory", "16g")   # Increase from 8g
```

---

## 📦 Managing Dependencies

### Freeze Working Versions

```bash
# Save exact package versions that work
pip freeze > requirements_frozen_$(date +%Y%m%d).txt
```

### Update Single Package

```bash
# Update TensorFlow (example)
pip install --upgrade tensorflow==2.18.0

# Test immediately
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 🎯 Best Practices

1. **Always save configuration** before system updates:
   ```bash
   ./config/save_nvidia_config.sh -action save
   ```

2. **Use virtual environments**, never install globally

3. **Freeze drivers** if TensorFlow works:
   ```bash
   ./config/save_nvidia_config.sh -action freeze
   ```

4. **Test after any change**:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

5. **Keep configuration snapshots** for rollback

---

## 📂 Project Structure

```
CropClassifier/
├── config/
│   ├── save_nvidia_config.sh           # Save/freeze NVIDIA config
│   └── gpu/
│       ├── setup_gpu.py                # Configure TensorFlow GPU
│       ├── check_gpu.py                # Full GPU diagnostic script
│       ├── test_cuda_direct.py         # Low-level CUDA test
│       └── check_cuda_init_error.sh    # Fix Docker GPU errors
├── preprocessing/
│   ├── datasources.py                  # Data download functions
│   ├── download_cdl_data.py            # CDL data download
│   ├── get_sentinel_tiles.py           # Sentinel-2 download
│   └── spark_session.py                # PySpark configuration
├── doc/
│   └── environment_setup.md            # This file
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker image definition
├── docker-compose.yml                  # Docker services
└── docker-run.sh                       # Docker helper script
```

---

## ⚙️ Configuration Files

**Key files to know:**

- `requirements.txt` - All Python dependencies with exact versions
- `config/save_nvidia_config.sh` - Manage NVIDIA/CUDA package freezing
- `config/gpu/setup_gpu.py` - Configure TensorFlow GPU memory growth
- `config/gpu/check_gpu.py` - Full GPU diagnostic (environment, nvidia-smi, TensorFlow)
- `config/gpu/test_cuda_direct.py` - Direct CUDA initialization test (low-level debugging)
- `config/gpu/check_cuda_init_error.sh` - Diagnose and fix Docker GPU errors (requires sudo)
- `preprocessing/spark_session.py` - Configure PySpark memory limits
- `Dockerfile` - Reproduces the exact tested environment
- `docker-compose.yml` - Docker services configuration

---

## 🚀 Quick Start Comparison

| Task | Virtual Environment | Docker |
|------|---------------------|--------|
| **Setup time** | ~5 minutes | ~15 minutes (first build) |
| **Isolation** | Same as host | Fully isolated |
| **GPU access** | Direct | Through nvidia-docker |
| **Portability** | Requires manual setup | Single command |
| **Updates** | Manual freeze needed | Dockerfile version controlled |
| **Best for** | Daily development | Reproducible experiments |

---

## 💡 Recommendations

- **Development**: Use virtual environment for faster iteration
- **Production/Sharing**: Use Docker for reproducibility
- **Both**: Use Docker for experiments, venv for development

You can switch between them anytime - both mount the same data directories.

---

## ⚠️ Known Issues: GPU Crashes with Large Datasets

### The Problem

When working with large geographic regions (large bounding boxes in `config/bbox_config.txt`), the GPU may crash with the error:

```
CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
```

This is a **low-level bug** in TensorFlow/CUDA that affects consumer GPUs (like RTX 4070) when processing complex models over extended periods. The crash typically occurs:

- After 30-60 minutes of continuous GPU usage
- During testing with multiple `start_day` iterations
- More frequently with larger batch sizes

### Affected Hardware

| GPU | VRAM | Typical Crash Point |
|-----|------|---------------------|
| RTX 4070 | 12 GB | After ~6-10 iterations |
| RTX 4060 Ti | 8 GB | May crash earlier |
| RTX 3090 | 24 GB | More stable, but not immune |

### Recovery After Crash

After a `CUDA_ERROR_LAUNCH_FAILED` crash:

1. **Docker restart is NOT enough** - The CUDA context becomes corrupted at the driver level
2. **You must restart the host machine** to recover GPU functionality
3. Run your command again - **checkpoints will resume automatically**

```bash
# After restarting your machine:
./docker-run.sh start
./docker-run.sh shell

# Inside container - will resume from checkpoint:
./cropclassifier.sh -action test -model your_model_name
```

### Mitigation Strategies

#### 1. Use Smaller Bounding Boxes

In `config/bbox_config.txt`, reduce the geographic region:

```bash
# Instead of a large region:
# min_lon=-95.5
# max_lon=-89.0

# Use smaller tiles:
min_lon=-93.0
max_lon=-91.0
```

#### 2. Reduce Batch Size

In `config/dataloader.txt`:

```bash
# Default (may crash on large datasets):
batch_size=1028

# More stable for large datasets:
batch_size=256
```

#### 3. Disable XLA Compilation (Recommended for Stability)

XLA (Accelerated Linear Algebra) compiles TensorFlow operations into optimized GPU kernels. While faster, these compiled kernels can cause crashes on consumer GPUs.

**Disable XLA for maximum stability:**

```bash
# Default behavior (XLA disabled for stability):
./cropclassifier.sh -action test -model your_model

# Enable XLA for maximum performance (may crash):
STABLE_GPU_MODE=0 ./cropclassifier.sh -action test -model inception1d_se_mixup_focal_attention_residual

```

| Mode | XLA | Performance | Stability |
|------|-----|-------------|-----------|
| `STABLE_GPU_MODE=1` (default) | Disabled | ~30% slower | ✅ Stable |
| `STABLE_GPU_MODE=0` | Enabled | Fastest | ⚠️ May crash |

**Technical details:**

When `STABLE_GPU_MODE=1` (default), the system:
- Disables XLA JIT compilation (`TF_XLA_FLAGS`, `TF_DISABLE_JIT`)
- Disables TensorFloat-32 (can cause numerical instability on RTX 40xx)
- Uses conservative GPU thread settings
- Enables eager execution to avoid complex graph compilation

#### 4. Trust the Checkpoint System

The testing pipeline automatically saves progress after each `start_day` iteration. If the GPU crashes:

1. Restart your machine
2. Run the same command
3. It will skip completed iterations and continue

```bash
# Example output after resuming:
📂 Found checkpoint file: results/test/checkpoints/...
✅ Resuming from checkpoint! Already completed start_days: [0, 15, 30, 45, 60]
⏭️  Skipping start_day=0 (already completed in checkpoint)
⏭️  Skipping start_day=15 (already completed in checkpoint)
...
🔄 Processing start_day=75 (6/15)
```

### Why This Happens

The root cause is a combination of:

1. **TensorFlow 2.17 + RTX 40xx (Ada Lovelace)**: Known compatibility issues with XLA
2. **Extended GPU usage**: Memory fragmentation and driver state corruption over time
3. **Complex model graphs**: The `inception1d_se_mixup_focal_attention_residual` model generates complex CUDA kernels
4. **Docker environment**: Additional layer that can exacerbate CUDA context issues

### Recommendations by Use Case

| Scenario | Recommended Settings |
|----------|---------------------|
| **Testing large regions** | `batch_size=256`, `STABLE_GPU_MODE=1` (default) |
| **Quick tests on small regions** | `batch_size=512`, can try `STABLE_GPU_MODE=0` |
| **Production/overnight runs** | `batch_size=256`, `STABLE_GPU_MODE=1`, expect possible restarts |
| **Training** | Usually more stable than testing; default settings work |

### Future Solutions

- Upgrading to newer TensorFlow versions may resolve XLA issues
- NVIDIA driver updates occasionally fix these problems
- Using enterprise GPUs (A100, V100) provides better stability
