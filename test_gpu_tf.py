#!/usr/bin/env python3
"""Test TensorFlow GPU functionality"""
import tensorflow as tf

print("=" * 60)
print("TensorFlow GPU Compatibility Test")
print("=" * 60)

# Check TensorFlow version and build info
print(f"\nTensorFlow Version: {tf.__version__}")
try:
    build_info = tf.sysconfig.get_build_info()
    print(f"CUDA Built: {build_info.get('cuda_version', 'N/A')}")
    print(f"cuDNN Built: {build_info.get('cudnn_version', 'N/A')}")
except:
    print("Build info not available")

# List GPUs
print("\n" + "-" * 60)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu.name} ({gpu.device_type})")

if not gpus:
    print("❌ No GPUs detected!")
    exit(1)

# Test GPU computation
print("\n" + "-" * 60)
print("Testing GPU computation...")
try:
    # Enable memory growth
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Simple matrix multiplication on GPU
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        result = c.numpy()
    
    print(f"Matrix multiplication result:\n{result}")
    print("\n✅ GPU computation successful!")
    print("✅ TensorFlow is working correctly with GPU")
    
except Exception as e:
    print(f"\n❌ GPU computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("=" * 60)
