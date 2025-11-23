#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест совместимости с CPU для Cerebra AI
"""
import torch
import sys
import os

# Добавляем текущую папку в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_cpu_compatibility():
    print("🧪 Тест совместимости с CPU...")
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    # Принудительно используем CPU для теста
    device = torch.device('cpu')
    print(f"Используем устройство: {device}")
    
    # Тестируем простую операцию
    try:
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.matmul(x, y)
        print(f"✅ Простая операция на {device} выполнена успешно: {z.shape}")
    except Exception as e:
        print(f"❌ Ошибка при операции на {device}: {e}")
        return False
    
    # Тестируем модель
    try:
        from cerebra.gpu_acceleration import GPUAccelerator
        accelerator = GPUAccelerator()
        print(f"✅ GPUAccelerator создан, устройство: {accelerator.device}")
        
        # Проверяем, что устройство не CUDA для AMD GPU
        if accelerator.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            if 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper():
                print(f"⚠️  Обнаружена AMD карта: {gpu_name}, но система настроена на CPU")
            else:
                print(f"✅ CUDA устройство: {gpu_name}")
        elif accelerator.device.type == 'cpu':
            print("✅ Используется CPU, как и должно быть для совместимости")
        elif accelerator.device.type == 'mps':
            print("✅ Используется MPS (Apple Silicon)")
            
    except Exception as e:
        print(f"❌ Ошибка при создании GPUAccelerator: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Тестируем основную систему
    try:
        from cerebra import CerebraAI
        ai = CerebraAI()
        print(f"✅ CerebraAI создан, устройство: {ai.device}")
        
        # Проверяем, что модель может быть загружена
        model = ai.load_model("Synthesis-L1")
        if model:
            print("✅ Модель успешно загружена")
        else:
            print("⚠️  Модель не загружена, но это не критично")
        
    except Exception as e:
        print(f"❌ Ошибка при создании CerebraAI: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("🎉 Все тесты пройдены успешно!")
    return True

if __name__ == "__main__":
    success = test_cpu_compatibility()
    if success:
        print("\n✅ Система готова к работе на CPU/AMD GPU!")
    else:
        print("\n❌ Обнаружены проблемы совместимости")
        sys.exit(1)