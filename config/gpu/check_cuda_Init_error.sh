#!/bin/bash
# Script to diagnose and fix CUDA_ERROR_UNKNOWN in Docker containers
# This script requires sudo privileges

set -e

echo "============================================================"
echo "CUDA_ERROR_UNKNOWN Diagnostic and Fix Script"
echo "============================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run with sudo"
    echo "Usage: sudo bash config/gpu/fix_cuda_error.sh"
    exit 1
fi

echo "📋 Step 1: Checking NVIDIA driver..."
nvidia-smi > /dev/null 2>&1 && echo "✅ NVIDIA driver is working" || echo "❌ NVIDIA driver not working"

echo ""
echo "📋 Step 2: Checking nvidia-container-toolkit..."
if command -v nvidia-container-cli &> /dev/null; then
    echo "✅ nvidia-container-cli found"
    nvidia-container-cli --version
else
    echo "❌ nvidia-container-cli not found"
fi

echo ""
echo "📋 Step 3: Checking for processes using GPU..."
GPU_PROCESSES=$(fuser /dev/nvidia* 2>&1 | grep -v "cannot" | wc -l)
if [ "$GPU_PROCESSES" -gt 0 ]; then
    echo "⚠️ Found processes using GPU:"
    fuser /dev/nvidia* 2>&1 | grep -v "cannot" || echo ""
else
    echo "✅ No processes blocking GPU"
fi

echo ""
echo "📋 Step 4: Restarting Docker service..."
systemctl restart docker
echo "✅ Docker service restarted"
sleep 3

echo ""
echo "📋 Step 5: Testing GPU access with simple container..."
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Basic GPU access test passed"
else
    echo "❌ Basic GPU access test failed"
fi

echo ""
echo "📋 Step 6: Checking nvidia-persistenced..."
if systemctl is-enabled nvidia-persistenced > /dev/null 2>&1; then
    echo "ℹ️ nvidia-persistenced status:"
    systemctl status nvidia-persistenced --no-pager -l | head -5 || true
else
    echo "ℹ️ nvidia-persistenced is masked (this is normal)"
fi

echo ""
echo "============================================================"
echo "Diagnostic complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Try running your container again:"
echo "   docker compose up -d cropclassifier"
echo "   docker compose exec cropclassifier python config/gpu/check_gpu.py"
echo ""
echo "2. If the problem persists, try restarting your system:"
echo "   sudo reboot"
echo ""
echo "3. If still not working, check Docker logs:"
echo "   docker compose logs cropclassifier"
echo ""


