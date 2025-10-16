import tensorflow as tf
import torch

from tensorflow.python.platform import build_info as tf_build_info

def detect_gpu():
    print("🔍 Checking for GPU...")
    
    print(torch.version.cuda)          # PyTorch CUDA version
    print(torch.cuda.is_available())   # Should be True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # GPU Detection
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {len(gpus)} available")
        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            print(f"\n GPU {i}:")
            print(f"  • Name: {details.get('device_name', 'Unknown')}")
            print(f"  • Compute Capability: {details.get('compute_capability', 'N/A')}")
    else:
        print("❌ No GPU detected. Running on CPU.")

    # TensorFlow version info
    print(f"\nTensorFlow version: {tf.__version__}")

    # CUDA/cuDNN info (check if keys exist)
    cuda_version = tf_build_info.build_info.get('cuda_version', None)
    cudnn_version = tf_build_info.build_info.get('cudnn_version', None)

    if cuda_version and cudnn_version:
        print(f"CUDA version: {cuda_version}")
        print(f"cuDNN version: {cudnn_version}")
    else:
        print("⚠️ CUDA/cuDNN version not found — likely running on CPU build of TensorFlow.")
