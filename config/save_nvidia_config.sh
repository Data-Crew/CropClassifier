#!/bin/bash

# Save CropClassifier environment configuration with optional freeze/unfreeze
#
# Purpose: Prevent system updates from breaking the working TensorFlow/CUDA setup.
# 
# TensorFlow is compiled for specific CUDA Toolkit versions. If apt upgrades remove
# cuda-12-x or update NVIDIA libraries, TensorFlow won't find the libraries it needs
# and GPU training will break. Freezing packages locks the current working configuration.
#
# The NVIDIA driver version (shown in nvidia-smi) indicates MAX CUDA support, but
# the actual CUDA Toolkit version (nvcc --version) is what TensorFlow uses.
#
# Usage: ./save_nvidia_config.sh [-action save|freeze|unfreeze]

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="environment_config_${TIMESTAMP}.txt"
ACTION="save"  # Default value

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -action)
            ACTION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-action save|freeze|unfreeze]"
            exit 1
            ;;
    esac
done

# Validate action parameter
if [[ "$ACTION" != "save" && "$ACTION" != "freeze" && "$ACTION" != "unfreeze" ]]; then
    echo "Error: -action must be 'save', 'freeze', or 'unfreeze'"
    echo "  save     = Just save config, don't change package status (default)"
    echo "  freeze   = Freeze packages to prevent updates"
    echo "  unfreeze = Unfreeze packages to allow updates"
    exit 1
fi

echo "=== CropClassifier Environment Configuration ==="
echo "Timestamp: $(date)"
echo "Action: $ACTION"
echo ""

# Detect installed NVIDIA/CUDA packages dynamically
echo "Detecting installed NVIDIA/CUDA packages..."
CUDA_PACKAGES=$(dpkg -l | grep -E '^ii' | grep -E 'cuda|nvidia' | awk '{print $2}')
CUDA_PACKAGES_ARRAY=($CUDA_PACKAGES)

echo "Found ${#CUDA_PACKAGES_ARRAY[@]} NVIDIA/CUDA packages installed"
echo ""

# Check current hold status
HELD_PACKAGES=$(apt-mark showhold | grep -E 'cuda|nvidia')
HELD_COUNT=$(echo "$HELD_PACKAGES" | grep -c '^' 2>/dev/null || echo 0)

# Write configuration to file
{
    echo "=== CropClassifier Environment Configuration - $(date) ==="
    echo ""
    echo "CUDA Toolkit:"
    nvcc --version | grep "release"
    echo ""
    echo "NVIDIA Driver and GPU:"
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
    echo ""
    echo "TensorFlow:"
    python -c "import tensorflow as tf; print('Version:', tf.__version__)" 2>/dev/null
    echo ""
    echo "GPU Detected:"
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" 2>/dev/null
    echo ""
    echo "NVIDIA/CUDA Packages Installed (${#CUDA_PACKAGES_ARRAY[@]} total):"
    echo "$CUDA_PACKAGES"
    echo ""
    echo "Currently Held Packages ($HELD_COUNT):"
    if [[ $HELD_COUNT -gt 0 ]]; then
        echo "$HELD_PACKAGES"
    else
        echo "  None - all packages can be updated"
    fi
    echo ""
    if [[ "$ACTION" == "freeze" ]]; then
        echo "Action Taken: FROZEN (held from updates)"
    elif [[ "$ACTION" == "unfreeze" ]]; then
        echo "Action Taken: UNFROZEN (released for updates)"
    else
        echo "Action Taken: NO CHANGE (status preserved)"
    fi
    echo ""
    echo "==============================================="
} > "${OUTPUT_FILE}"

echo "✓ Configuration saved in: ${OUTPUT_FILE}"
echo ""

# Freeze packages if requested
if [[ "$ACTION" == "freeze" ]]; then
    echo "Freezing ALL installed NVIDIA/CUDA packages..."
    echo ""
    
    FROZEN_COUNT=0
    for pkg in "${CUDA_PACKAGES_ARRAY[@]}"; do
        echo "  Holding: $pkg"
        sudo apt-mark hold "$pkg" 2>/dev/null
        if [[ $? -eq 0 ]]; then
            ((FROZEN_COUNT++))
        fi
    done
    
    echo ""
    echo "✓ Frozen $FROZEN_COUNT packages"
    echo ""
    echo "Packages currently held:"
    apt-mark showhold | grep -E 'cuda|nvidia'
    
    # Append freeze info to file
    {
        echo ""
        echo "FROZEN PACKAGES ($(date)):"
        apt-mark showhold | grep -E 'cuda|nvidia'
    } >> "${OUTPUT_FILE}"
    
elif [[ "$ACTION" == "unfreeze" ]]; then
    echo "Unfreezing ALL NVIDIA/CUDA packages..."
    echo ""
    
    if [[ $HELD_COUNT -eq 0 ]]; then
        echo "No packages were frozen. Nothing to unfreeze."
    else
        UNFROZEN_COUNT=0
        HELD_PACKAGES_ARRAY=($HELD_PACKAGES)
        for pkg in "${HELD_PACKAGES_ARRAY[@]}"; do
            echo "  Releasing: $pkg"
            sudo apt-mark unhold "$pkg" 2>/dev/null
            if [[ $? -eq 0 ]]; then
                ((UNFROZEN_COUNT++))
            fi
        done
        
        echo ""
        echo "✓ Unfrozen $UNFROZEN_COUNT packages"
        echo ""
        echo "Remaining held packages:"
        REMAINING=$(apt-mark showhold | grep -E 'cuda|nvidia')
        if [[ -z "$REMAINING" ]]; then
            echo "  None - all NVIDIA/CUDA packages are now free to update"
        else
            echo "$REMAINING"
        fi
        
        # Append unfreeze info to file
        {
            echo ""
            echo "UNFROZEN PACKAGES ($(date)):"
            echo "All NVIDIA/CUDA packages released for updates"
        } >> "${OUTPUT_FILE}"
    fi
fi

echo ""
echo "File content:"
echo "----------------------------------------"
cat "${OUTPUT_FILE}"
echo "----------------------------------------"

if [[ "$ACTION" == "save" ]]; then
    echo ""
    if [[ $HELD_COUNT -gt 0 ]]; then
        echo "Current status: $HELD_COUNT packages are FROZEN"
        echo ""
        echo "To unfreeze and allow updates, run:"
        echo "  ./save_nvidia_config.sh -action unfreeze"
    else
        echo "Current status: All packages are FREE to update"
        echo ""
        echo "To freeze packages and prevent updates, run:"
        echo "  ./save_nvidia_config.sh -action freeze"
    fi
fi