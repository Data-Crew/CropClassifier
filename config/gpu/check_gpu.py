#!/usr/bin/env python3
"""
GPU Diagnostic Script

Quick diagnostic tool to check GPU availability and TensorFlow GPU configuration.
Useful for troubleshooting GPU detection issues after container crashes or restarts.

Usage:
    python config/gpu/check_gpu.py
    or
    ./config/gpu/check_gpu.py

This script checks:
    - Environment variables (NVIDIA_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, etc.)
    - nvidia-smi availability
    - TensorFlow GPU detection
    - CUDA library paths

If GPU is not detected but nvidia-smi works, try restarting the Docker container.
"""
import os
import sys
from pathlib import Path

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("GPU DIAGNOSTIC CHECK")
print("=" * 60)

# Check environment variables
print("\n📋 Environment Variables:")
print(f"  NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:100]}...")

# Check nvidia-smi
print("\n🔍 Checking nvidia-smi...")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ nvidia-smi works!")
        # Extract GPU name
        for line in result.stdout.split('\n'):
            if 'GeForce' in line or 'RTX' in line or 'Tesla' in line:
                print(f"  {line.strip()}")
    else:
        print("❌ nvidia-smi failed")
        print(result.stderr)
except Exception as e:
    print(f"❌ Error running nvidia-smi: {e}")

# Check TensorFlow
print("\n🔍 Checking TensorFlow...")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Check CUDA build
    print(f"  CUDA built: {tf.test.is_built_with_cuda()}")
    
    # Check GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  Physical GPUs found: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"    GPU {i}: {gpu.name}")
        print(f"      Details: {gpu}")
    
    # Check logical devices
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(f"  Logical GPUs: {len(logical_gpus)}")
    
    # Try to configure GPU
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth configured successfully")
        except Exception as e:
            print(f"⚠️  Warning configuring GPU: {e}")
    else:
        print("⚠️  No GPUs detected by TensorFlow")
        print("\n💡 Troubleshooting:")
        print("  1. Restart Docker container:")
        print("     docker restart cropclassifier-main")
        print("     # Or if using docker-compose:")
        print("     docker-compose restart cropclassifier")
        print("  2. Check Docker GPU access:")
        print("     docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi")
        print("  3. Verify nvidia-container-toolkit is installed:")
        print("     dpkg -l | grep nvidia-container-toolkit")
        print("  4. If error shows 'CUDA_ERROR_UNKNOWN', restart is usually needed")
        
except ImportError:
    print("❌ TensorFlow not installed")
except Exception as e:
    print(f"❌ Error checking TensorFlow: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

