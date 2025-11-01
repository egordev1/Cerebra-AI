import torch

def print_system_info():
    print("=" * 50)
    print("🧠 Cerebra AI - System Information")
    print("=" * 50)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 50)