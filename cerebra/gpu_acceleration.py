import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """Класс для ускорения работы моделей с помощью GPU"""
    
    def __init__(self):
        self.device = self._get_device()
        # Убираем GradScaler так как он специфичен для CUDA
        self.scaler = None
        logger.info(f"Используется устройство: {self.device}")
    
    def _get_device(self):
        """Определяет доступное устройство (GPU или CPU)"""
        # Проверяем доступность MPS (Metal Performance Shaders для Apple Silicon) или CUDA
        # или просто CPU
        if torch.cuda.is_available():
            # CUDA доступна, проверим поддержку MPS для Apple Silicon
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Используется MPS (Apple Silicon)")
            else:
                # Проверим, может быть это AMD GPU
                try:
                    # Попробуем получить информацию о CUDA устройстве
                    device = torch.device('cuda')
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"Найдено GPU устройство: {gpu_name}")
                    # Для AMD GPU используем CPU, так как PyTorch может не поддерживать все AMD GPU
                    # Проверим, является ли это AMD GPU
                    if 'AMD' in gpu_name.upper() or 'RADEON' in gpu_name.upper():
                        logger.warning(f"Найдена AMD карта: {gpu_name}, используется CPU для совместимости")
                        device = torch.device('cpu')
                    else:
                        logger.info(f"CUDA устройство: {gpu_name}")
                        logger.info(f"Количество CUDA устройств: {torch.cuda.device_count()}")
                        logger.info(f"Версия CUDA: {torch.version.cuda}")
                except:
                    # Если ошибка при получении информации о GPU, используем CPU
                    device = torch.device('cpu')
                    logger.warning("Ошибка при определении GPU, используется CPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Используется MPS (Apple Silicon)")
        else:
            device = torch.device('cpu')
            logger.info("GPU недоступна, используется CPU")
        return device
    
    def move_to_device(self, model):
        """Перемещает модель на GPU если доступно"""
        model = model.to(self.device)
        return model
    
    def move_tensor_to_device(self, tensor):
        """Перемещает тензор на GPU если доступно"""
        tensor = tensor.to(self.device)
        return tensor
    
    def autocast_context(self):
        """Контекстный менеджер для автоматического определения типа данных"""
        # Убираем autocast так как он специфичен для CUDA
        # Возвращаем пустой контекстный менеджер для всех устройств
        from contextlib import nullcontext
        return nullcontext()
    
    def scale_loss(self, loss):
        """Масштабирует loss для GPU или возвращает как есть для CPU"""
        # Убираем масштабирование так как scaler больше не используется
        return loss
    
    def update_scaler(self, optimizer):
        """Обновляет scaler для GPU или делает шаг оптимизатора для CPU"""
        # Убираем scaler так как он больше не используется
        optimizer.step()
    
    def zero_grad(self, optimizer):
        """Обнуляет градиенты"""
        optimizer.zero_grad()
    
    def is_gpu_available(self):
        """Проверяет доступность GPU"""
        # Возвращаем True только для CUDA и MPS, но не для CPU
        return self.device.type in ['cuda', 'mps']
    
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
        elif self.device.type == 'mps':
            try:
                # Для MPS используем специфичные функции
                import subprocess
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                      capture_output=True, text=True)
                # Возвращаем базовую информацию
                return {
                    'allocated': 0,
                    'reserved': 0,
                    'total': 0,
                    'available': 0
                }
            except:
                return {
                    'allocated': 0,
                    'reserved': 0,
                    'total': 0,
                    'available': 0
                }
        else:
            return None