import tensorflow as tf
import os
import sys


def disable_xla_jit():
    """
    Disable XLA JIT compilation which can cause 'unspecified launch failure' errors.
    XLA generates complex CUDA kernels that sometimes fail on consumer GPUs.
    
    Call this BEFORE any TensorFlow operations.
    """
    # Disable XLA JIT completely
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    os.environ['TF_XLA_ENABLE_XLA_DEVICES'] = '0'
    
    # Alternative: disable via TensorFlow config
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass  # May fail if called too late
    
    print("🛑 XLA JIT compilation disabled for stability.")


def configure_stable_gpu():
    """
    Configure GPU with conservative settings for maximum stability.
    Use this if you're experiencing CUDA crashes.
    """
    # Disable TensorFloat-32 (can cause numerical instability on RTX 30xx/40xx)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    print("🔧 TensorFloat-32 disabled for stability.")
    
    # Set memory growth via environment variable (backup method)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Reduce GPU thread complexity
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1'
    
    print("🔧 GPU configured with conservative settings.")


def _check_cuda_state():
    """
    Check if CUDA is in a valid state by attempting to list physical devices.
    Returns True if CUDA is accessible, False otherwise.
    """
    try:
        # Try to list physical devices - this will fail if CUDA is in a bad state
        gpus = tf.config.list_physical_devices('GPU')
        return True
    except Exception as e:
        error_msg = str(e).lower()
        # Check for common CUDA initialization errors
        if 'cuda_error_unknown' in error_msg or 'cuinit' in error_msg:
            return False
        # For other errors, assume CUDA is accessible (might be a different issue)
        return True


def reset_cuda_context():
    """
    Attempt to reset CUDA context by clearing TensorFlow session.
    This can help recover from CUDA_ERROR_UNKNOWN states.
    """
    try:
        # Clear any existing TensorFlow sessions
        tf.keras.backend.clear_session()
        # Try to reset memory stats if available
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass
        return True
    except Exception as e:
        print(f"⚠️ Warning: Could not reset CUDA context: {e}")
        return False


# Initial config
def check_GPU_config():
    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU devices found. Ensure your system recognizes the GPU.")
    else:
        try:
            # Limit TensorFlow to use only the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"Configured TensorFlow to use GPU: {gpus[0].name}")

            # Enable dynamic memory growth on the GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("Memory growth enabled for the first GPU.")

            # Optional: Display additional GPU configuration details
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")

        except RuntimeError as e:
            print(f"RuntimeError during GPU setup: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Further GPU diagnostics
    print("TensorFlow version:", tf.__version__)
    print("CUDA device detected:", tf.test.is_built_with_cuda())
    print("GPU availability:", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    return gpus

# GPU configuration for TensorFlow to enable dynamic memory allocation.
def configure_tensorflow_gpu(stable_mode: bool = True) -> None:
    """
    Configures TensorFlow to use GPU memory efficiently, allowing dynamic memory growth.

    This function should be called before initializing any TensorFlow models or datasets
    to ensure that memory usage is optimized for systems with available GPUs.
    
    Args:
        stable_mode: If True (default), disables XLA JIT and uses conservative GPU settings
                     to prevent 'CUDA_ERROR_LAUNCH_FAILED' crashes. Set to False for 
                     maximum performance if you don't experience crashes.
    
    Handles CUDA_ERROR_UNKNOWN by attempting to reset the CUDA context.
    """
    # === STABILITY SETTINGS ===
    # These must be set BEFORE any TensorFlow operations
    if stable_mode:
        disable_xla_jit()
        configure_stable_gpu()
    
    # Check if CUDA is in a valid state before proceeding
    if not _check_cuda_state():
        print("⚠️ CUDA appears to be in an invalid state (CUDA_ERROR_UNKNOWN).")
        print("🔄 Attempting to reset CUDA context...")
        if reset_cuda_context():
            print("✅ CUDA context reset. Retrying GPU detection...")
        else:
            print("❌ Could not reset CUDA context. You may need to restart the Docker container.")
            print("💡 Try: docker compose restart cropclassifier")
            return
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Enabled memory growth for {len(gpus)} GPU(s).")
            except RuntimeError as e:
                # RuntimeError usually means GPU was already configured
                # This is OK, just inform the user
                if "already been set" in str(e) or "already configured" in str(e).lower():
                    print(f"ℹ️ GPU memory growth already configured.")
                else:
                    print(f"❌ Failed to set memory growth: {e}")
                    raise
        else:
            print("⚠️ No GPUs found. Running on CPU.")
    except Exception as e:
        error_msg = str(e).lower()
        if 'cuda_error_unknown' in error_msg or 'cuinit' in error_msg:
            print(f"❌ CUDA initialization error: {e}")
            print("💡 This usually means CUDA context is corrupted.")
            print("💡 Solution: Restart the Docker container:")
            print("   docker compose restart cropclassifier")
            print("   # Or from inside container:")
            print("   exit  # then restart container")
            raise
        else:
            print(f"❌ Unexpected error configuring GPU: {e}")
            raise
    
    print("✅ GPU configuration completed!")