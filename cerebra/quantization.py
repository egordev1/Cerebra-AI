import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, fuse_modules
import logging

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Класс для квантования моделей с целью ускорения и уменьшения памяти"""
    
    def __init__(self):
        self.quantization_methods = {
            'dynamic': self.dynamic_quantization,
            'static': self.static_quantization,
            'qat': self.quantization_aware_training
        }
    
    def dynamic_quantization(self, model, dtype=torch.qint8):
        """
        Динамическое квантование - квантует веса во время выполнения
        """
        logger.info("Применение динамического квантования...")
        
        # Определяем модули для квантования
        quantizable_modules = {
            torch.nn.Linear,
            torch.nn.LSTM,
            torch.nn.GRU,
            torch.nn.RNN
        }
        
        # Применяем динамическое квантование
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec=quantizable_modules,
            dtype=dtype
        )
        
        logger.info(f"Модель квантована динамически. Тип: {dtype}")
        return quantized_model
    
    def static_quantization(self, model, calibration_data_loader=None):
        """
        Статическое квантование - требует калибровочных данных
        """
        logger.info("Подготовка к статическому квантования...")
        
        # Устанавливаем модель в режим оценки
        model.eval()
        
        # Конфигурация квантования
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Фьюзим модули для лучшей производительности
        self._fuse_model_modules(model)
        
        # Применяем подготовку квантования
        torch.quantization.prepare(model, inplace=True)
        
        # Если предоставлены калибровочные данные, выполняем калибровку
        if calibration_data_loader is not None:
            logger.info("Выполнение калибровки для статического квантования...")
            with torch.no_grad():
                for data in calibration_data_loader:
                    model(data)
        
        # Завершаем квантование
        torch.quantization.convert(model, inplace=True)
        
        logger.info("Статическое квантование завершено")
        return model
    
    def quantization_aware_training(self, model, dtype=torch.qint8):
        """
        Квантование с учетом обучения - симулирует квантование во время обучения
        """
        logger.info("Подготовка квантования с учетом обучения...")
        
        # Устанавливаем модель в режим обучения
        model.train()
        
        # Конфигурация QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Фьюзим модули
        self._fuse_model_modules(model)
        
        # Подготавливаем модель к QAT
        torch.quantization.prepare_qat(model, inplace=True)
        
        logger.info("Квантование с учетом обучения подготовлено")
        return model
    
    def _fuse_model_modules(self, model):
        """
        Фьюзит модули для лучшей производительности квантования
        """
        # Пример фьюза для типичных слоев трансформера
        for module in model.modules():
            if isinstance(module, nn.Sequential):
                # Пытаемся фьюзить линейные слои с активациями
                try:
                    if len(module) >= 2:
                        # Проверяем, можно ли фьюзить Linear + ReLU или Linear + GELU
                        for i in range(len(module) - 1):
                            if isinstance(module[i], nn.Linear) and isinstance(module[i+1], (nn.ReLU, nn.GELU)):
                                try:
                                    fuse_modules(module, [str(i), str(i+1)], inplace=True)
                                except:
                                    continue
                except:
                    continue
    
    def apply_quantization(self, model, method='dynamic', **kwargs):
        """
        Применяет указанное квантование к модели
        """
        if method not in self.quantization_methods:
            raise ValueError(f"Неизвестный метод квантования: {method}")
        
        quantizer = self.quantization_methods[method]
        return quantizer(model, **kwargs)
    
    def calculate_model_size(self, model):
        """
        Рассчитывает размер модели в байтах
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def calculate_compression_ratio(self, original_model, quantized_model):
        """
        Рассчитывает степень сжатия модели
        """
        original_size = self.calculate_model_size(original_model)
        quantized_size = self.calculate_model_size(quantized_model)
        
        if original_size > 0:
            ratio = original_size / quantized_size
            return ratio, original_size, quantized_size
        else:
            return 1.0, 0, 0

class AdvancedQuantizer(ModelQuantizer):
    """
    Расширенный класс квантования с дополнительными методами
    """
    
    def mixed_precision_quantization(self, model):
        """
        Квантует разные части модели с разной точностью
        """
        logger.info("Применение смешанной точности квантования...")
        
        # Для трансформерных моделей квантует attention слои с меньшей точностью
        # а feed-forward слои с большей точностью
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Определяем тип линейного слоя по имени
                if 'attn' in name.lower() or 'attention' in name.lower():
                    # Attention слои квантуются с int8
                    module.qconfig = torch.quantization.default_qconfig
                elif 'ff' in name.lower() or 'feed_forward' in name.lower():
                    # Feed-forward слои квантуются с int4
                    module.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        
        # Применяем квантование
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        
        return model