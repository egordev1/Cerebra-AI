import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import logging

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """Класс для ускорения работы моделей с помощью GPU"""
    
    def __init__(self):
        self.device = self._get_device()
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        logger.info(f"Используется устройство: {self.device}")
    
    def _get_device(self):
        """Определяет доступное устройство (GPU или CPU)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"CUDA устройство: {torch.cuda.get_device_name(0)}")
            logger.info(f"Количество CUDA устройств: {torch.cuda.device_count()}")
            logger.info(f"Версия CUDA: {torch.version.cuda}")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA недоступна, используется CPU")
        return device
    
    def move_to_device(self, model):
        """Перемещает модель на GPU если доступно"""
        if self.device.type == 'cuda':
            model = model.to(self.device)
        return model
    
    def move_tensor_to_device(self, tensor):
        """Перемещает тензор на GPU если доступно"""
        if self.device.type == 'cuda':
            tensor = tensor.to(self.device)
        return tensor
    
    def autocast_context(self):
        """Контекстный менеджер для автоматического определения типа данных"""
        if self.device.type == 'cuda':
            return autocast()
        else:
            # Возвращаем пустой контекстный менеджер для CPU
            from contextlib import nullcontext
            return nullcontext()
    
    def scale_loss(self, loss):
        """Масштабирует loss для GPU или возвращает как есть для CPU"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        else:
            return loss
    
    def update_scaler(self, optimizer):
        """Обновляет scaler для GPU или делает шаг оптимизатора для CPU"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def zero_grad(self, optimizer):
        """Обнуляет градиенты"""
        optimizer.zero_grad()
    
    def is_gpu_available(self):
        """Проверяет доступность GPU"""
        return self.device.type == 'cuda'
    
    def get_memory_info(self):
        """Получает информацию о памяти GPU"""
        if self.device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            memory_total = torch.cuda.get_device_properties(self.device).total_memory
            return {
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'total': memory_total,
                'available': memory_total - memory_allocated
            }
        else:
            return None