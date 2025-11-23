import torch

def print_system_info():
    print("=" * 50)
    print("🧠 Cerebra AI - System Information")
    print("=" * 50)
    print(f"PyTorch: {torch.__version__}")
    
    # Проверяем доступность различных устройств
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    print(f"CUDA: {cuda_available}")
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            # Проверяем, является ли это AMD GPU
            if 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper():
                print(f"GPU: {gpu_name} (AMD GPU - будет использоваться CPU для совместимости)")
            else:
                print(f"GPU: {gpu_name}")
        except:
            print("GPU: Доступно, но не определено")
    elif mps_available:
        print("MPS: Доступно (Apple Silicon)")
    else:
        print("GPU: Недоступно, используется CPU")
    
    print("=" * 50)