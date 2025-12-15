#!/usr/bin/env python3
"""
setup_gpu.py

Script to configure TensorFlow GPU settings.
This script should be called once before running training or inference scripts.

Usage:
    python config/gpu/setup_gpu.py
    or
    ./config/gpu/setup_gpu.py
"""

import sys
import os
from pathlib import Path

def setup_project_path():
    """Add project root to Python path to enable imports."""
    # Get the project root (two levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    sys.path.insert(0, str(project_root))
    return project_root

def main():
    """Main function to configure GPU settings."""
    try:
        # Setup project path
        project_root = setup_project_path()
        print(f"📁 Project root: {project_root}")
        
        # Check STABLE_GPU_MODE environment variable
        # Set STABLE_GPU_MODE=0 to disable stability settings and use full XLA performance
        stable_mode = os.environ.get('STABLE_GPU_MODE', '1') == '1'
        
        # Import GPU configuration function
        from config.gpu.gpu_utils import configure_tensorflow_gpu
        
        print("🔧 Configuring TensorFlow GPU...")
        configure_tensorflow_gpu(stable_mode=stable_mode)
        print("✅ GPU configuration completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this script from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 