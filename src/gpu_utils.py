import torch

def CheckGPU():
    print("=" * 50)
    print("🖥️  GPU INFORMATION")
    print("=" * 50)

    if torch.cuda.is_available():
        device_id = 0
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device_id)
        
        print(f"✅ GPU Detected         : {props.name}")
        print(f"   • Device ID          : {device_id}")
        print(f"   • Compute Capability : {props.major}.{props.minor}")
        print(f"   • Multiprocessors    : {props.multi_processor_count}")
        print(f"   • Total VRAM         : {props.total_memory / (1024 ** 3):.2f} GB")

        # Optional: check allocated and reserved memory
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
        print(f"   • VRAM Allocated     : {allocated:.2f} GB")
        print(f"   • VRAM Reserved      : {reserved:.2f} GB")
        print(f"   • Active Device      : {device}")

    else:
        print("❌ No GPU detected.")
        print(f"   • Active Device      : CPU")

    print("=" * 50)

def CheckGPUBrief():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        cudnn_ver = torch.backends.cudnn.version()
        print(f"🟢 GPU: {props.name} | 💾 VRAM: {vram_gb:.2f} GB")
        print(f"🧠 PyTorch: {torch.__version__} | 🧰 cuDNN: {cudnn_ver}")
    else:
        print("🔴 No GPU detected — using CPU")
        print(f"🧠 PyTorch: {torch.__version__} | 🧰 cuDNN: N/A")
    

def CheckCUDA():
    print("\n" + "=" * 50)
    print("⚡ CUDA / PYTORCH INFORMATION")
    print("=" * 50)

    cuda_available = torch.cuda.is_available()
    print(f"{'✅' if cuda_available else '❌'} CUDA Available       : {cuda_available}")
    print(f"   • PyTorch CUDA Ver.  : {torch.version.cuda}")
    print(f"   • PyTorch Version    : {torch.__version__}")

    if cuda_available:
        print(f"✅ cuDNN Version        : {torch.backends.cudnn.version()}")
        print(f"   • CUDA Device Count  : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   • Device {i} Name     : {torch.cuda.get_device_name(i)}")
    else:
        print("❌ cuDNN Version        : Not available (No GPU)")

    print("=" * 50)
