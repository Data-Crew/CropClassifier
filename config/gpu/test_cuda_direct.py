#!/usr/bin/env python3
"""
Direct CUDA test script to diagnose CUDA_ERROR_UNKNOWN
"""
import ctypes
import os
import sys

def test_cuda():
    """Test CUDA initialization directly"""
    print("=" * 60)
    print("Direct CUDA Test")
    print("=" * 60)
    
    # Try to load libcuda
    try:
        libcuda = ctypes.CDLL('libcuda.so.1')
        print("✅ libcuda.so.1 loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load libcuda.so.1: {e}")
        return False
    
    # Check if cuInit exists
    if not hasattr(libcuda, 'cuInit'):
        print("❌ cuInit not found in libcuda")
        return False
    
    # Try cuInit
    print("\n🔍 Testing cuInit...")
    result = libcuda.cuInit(0)
    
    if result == 0:
        print("✅ cuInit succeeded!")
        
        # Try to get device count
        try:
            count = ctypes.c_int()
            result2 = libcuda.cuDeviceGetCount(ctypes.byref(count))
            if result2 == 0:
                print(f"✅ Found {count.value} GPU(s)")
                return True
            else:
                print(f"⚠️ cuDeviceGetCount returned: {result2}")
        except Exception as e:
            print(f"⚠️ Error getting device count: {e}")
        return True
    else:
        print(f"❌ cuInit failed with error code: {result}")
        print("   Error codes: 0=SUCCESS, 1=INVALID_VALUE, 999=UNKNOWN_ERROR")
        
        # Check environment
        print("\n📋 Environment:")
        print(f"   NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'NOT SET')}")
        print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:100]}")
        
        # Check devices
        print("\n📋 Checking /dev/nvidia* devices:")
        for dev in ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm']:
            exists = os.path.exists(dev)
            readable = os.access(dev, os.R_OK) if exists else False
            print(f"   {dev}: exists={exists}, readable={readable}")
        
        return False

if __name__ == "__main__":
    success = test_cuda()
    sys.exit(0 if success else 1)


