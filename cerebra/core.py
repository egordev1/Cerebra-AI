"""
Ядро системы Cerebra AI
Файл: core.py - Главный контроллер AI системы, управляет моделями, обучением, чатом
"""
import torch
import os
import sys
import io

# Установка UTF-8 кодировки для Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass  # Если уже установлено

from .logger_config import logger
from .gpu_acceleration import GPUAccelerator
from .quantization import ModelQuantizer, AdvancedQuantizer
from .plugin_system import create_default_plugin_manager, PluginManager

class CerebraAI:
    def __init__(self):
        self.name = "CerebraAI"
        self.version = "2.0.0"
        self.active_model = None
        
        # Инициализация ускорения и квантования
        self.gpu_accelerator = GPUAccelerator()
        self.quantizer = ModelQuantizer()
        self.advanced_quantizer = AdvancedQuantizer()
        
        # Инициализация плагинов
        self.plugin_manager = create_default_plugin_manager()
        
        # Определение устройства через GPU ускоритель
        self.device = self.gpu_accelerator.device
        
        logger.info(f"🧠 Запущена {self.name} AI System v{self.version}")
        logger.info(f"📊 Устройство: {self.device}")
        print(f"🧠 Запущена {self.name} AI System v{self.version}")
        print(f"📊 Устройство: {self.device}")
        
        # Информация о GPU если доступно
        if self.gpu_accelerator.is_gpu_available():
            memory_info = self.gpu_accelerator.get_memory_info()
            if memory_info:
                logger.info(f"💾 GPU Память - Всего: {memory_info['total'] / 1024**3:.2f}GB, "
                           f"Использовано: {memory_info['allocated'] / 1024**3:.2f}GB")
    
    def load_model(self, model_name="Synthesis-L1", quantize=False, quantization_method='dynamic'):
        try:
            logger.info(f"Загрузка модели {model_name} на устройство {self.device}...")
            print(f"\n📦 Загрузка модели {model_name}...")
            
            if model_name == "Synthesis-L1":
                from .models.main_model import SynthesisL1
                logger.info(f"Инициализация {model_name}...")
                self.active_model = SynthesisL1(use_gpt=True)
                
                # Перемещаем модель на устройство
                if hasattr(self.active_model, 'gpt_model'):
                    self.active_model.gpt_model = self.active_model.gpt_model.to(self.device)
                self.active_model = self.active_model.to(self.device)
                logger.info(f"✅ Загружена модель: {model_name} на {self.device}")
                print(f"✅ Загружена модель: {model_name}")
                
            elif model_name == "Synthesis-L2":
                from .models.advanced_transformer import SynthesisL2
                logger.info(f"Инициализация {model_name}...")
                # Используем torch.no_grad() и eval режим для экономии памяти при инициализации
                with torch.no_grad():
                    self.active_model = SynthesisL2()
                    # Переводим в eval режим сразу после создания
                    self.active_model.eval()
                    # Перемещаем на устройство
                    self.active_model = self.active_model.to(self.device)
                logger.info(f"✅ Загружена модель: {model_name} на {self.device}")
                print(f"✅ Загружена модель: {model_name}")
                
            elif model_name == "Synthesis-L3":
                from .models.advanced_transformer import SynthesisL3
                logger.info(f"Инициализация {model_name}...")
                # Используем torch.no_grad() и eval режим для экономии памяти при инициализации
                with torch.no_grad():
                    self.active_model = SynthesisL3()
                    # Переводим в eval режим сразу после создания
                    self.active_model.eval()
                    # Перемещаем на устройство
                    self.active_model = self.active_model.to(self.device)
                logger.info(f"✅ Загружена модель: {model_name} на {self.device}")
                print(f"✅ Загружена модель: {model_name}")
                
            else:
                logger.error(f"Модель {model_name} не найдена")
                print(f"❌ Модель {model_name} не найдена")
                return None
            
            # Применяем квантование если нужно (это поможет уменьшить потребление памяти)
            if quantize:
                logger.info(f"Применение {quantization_method} квантования к модели...")
                # Очищаем кэш перед квантованием
                import gc
                gc.collect()
                self.active_model = self.quantizer.apply_quantization(self.active_model, method=quantization_method)
                logger.info(f"✅ Квантование {quantization_method} применено")
                print(f"✅ Квантование {quantization_method} применено")
            
            logger.info(f"Модель {model_name} успешно загружена")
            print(f"✅ Модель {model_name} готова к использованию")
            return self.active_model
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"Ошибка нехватки видеопамяти при загрузке модели {model_name}: {e}")
            print(f"❌ Ошибка нехватки видеопамяти при загрузке {model_name}")
            # Очищаем кэш CUDA если доступна
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except MemoryError as e:
            logger.error(f"Ошибка нехватки оперативной памяти при загрузке модели {model_name}: {e}")
            print(f"❌ Ошибка нехватки оперативной памяти при загрузке {model_name}")
            # Пытаемся освободить память
            import gc
            gc.collect()
            return None
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}", exc_info=True)
            print(f"❌ Ошибка при загрузке модели {model_name}: {e}")
            
            # Fallback на CPU если CUDA недоступна
            if self.device.type == 'cuda':
                logger.warning("Попытка загрузить модель на CPU вместо CUDA")
                self.device = torch.device('cpu')
                try:
                    if model_name == "Synthesis-L1":
                        from .models.main_model import SynthesisL1
                        self.active_model = SynthesisL1().to(self.device)
                    elif model_name == "Synthesis-L2":
                        from .models.advanced_transformer import SynthesisL2
                        # Используем torch.no_grad() и eval режим для экономии памяти
                        with torch.no_grad():
                            self.active_model = SynthesisL2()
                            self.active_model.eval()
                        self.active_model = self.active_model.to(self.device)
                    elif model_name == "Synthesis-L3":
                        from .models.advanced_transformer import SynthesisL3
                        # Используем torch.no_grad() и eval режим для экономии памяти
                        with torch.no_grad():
                            self.active_model = SynthesisL3()
                            self.active_model.eval()
                        self.active_model = self.active_model.to(self.device)
                    logger.info(f"✅ Модель загружена на CPU")
                    print(f"✅ Загружена модель: {model_name} на CPU (fallback)")
                    return self.active_model
                except Exception as e2:
                    logger.error(f"Ошибка при загрузке на CPU: {e2}", exc_info=True)
                    print(f"❌ Ошибка при загрузке на CPU: {e2}")
            return None
    
    def chat(self, message, use_web_search=False, use_plugins=True):
        """
        Общение с ИИ
        
        Args:
            message: Сообщение пользователя
            use_web_search: Использовать веб-поиск для ответа
            use_plugins: Использовать плагины для обработки запроса
        """
        if not self.active_model:
            logger.warning("Попытка чата без загруженной модели")
            return "⚠️ Сначала загрузите модель: ai.load_model()"
        
        logger.debug(f"Обработка сообщения: {message}")
        
        # Выполняем хук предварительной обработки
        self.plugin_manager.execute_hook('pre_process', message)
        
        # Проверяем, нужно ли использовать плагины
        if use_plugins:
            # Проверяем, не является ли запрос командой для плагина
            response = self._process_plugin_request(message)
            if response:
                self.plugin_manager.execute_hook('post_process', response)
                return response
        
        # Веб-поиск если нужно
        if use_web_search:
            try:
                from .web_search import web_searcher
                web_answer = web_searcher.get_answer_from_web(message)
                if web_answer:
                    logger.info("Использован ответ из веб-поиска")
                    # Сохраняем обмен для обучения
                    if hasattr(self.active_model, 'dialogue_collector') and self.active_model.dialogue_collector:
                        self.active_model.dialogue_collector.add_exchange(message, web_answer)
                    self.plugin_manager.execute_hook('post_process', web_answer)
                    return web_answer
            except Exception as e:
                logger.warning(f"Ошибка веб-поиска: {e}")
        
        # Генерация ответа моделью
        response = self.active_model.process(message)
        logger.debug(f"Получен ответ: {response}")
        
        # Сохраняем диалог для автоматического обучения
        if hasattr(self.active_model, 'dialogue_collector') and self.active_model.dialogue_collector:
            self.active_model.dialogue_collector.add_exchange(message, response)
        
        # Выполняем хук пост-обработки
        self.plugin_manager.execute_hook('post_process', response)
        
        return response
    
    def _process_plugin_request(self, message):
        """Обработка запросов к плагинам"""
        message_lower = message.lower()
        
        # Проверяем, содержит ли сообщение команду плагина
        if message_lower.startswith('!'):
            parts = message.split(' ', 1)
            command = parts[0][1:]  # Убираем !
            
            if command == 'calc' or 'калькулятор' in message_lower or 'посчитай' in message_lower:
                # Плагин калькулятора
                try:
                    expression = parts[1] if len(parts) > 1 else ''
                    return self.plugin_manager.execute_plugin('calculator', expression)
                except:
                    return "Укажите выражение для вычисления: !calc 2+2"
            
            elif command == 'time' or 'время' in message_lower:
                # Плагин времени
                return self.plugin_manager.execute_plugin('datetime', 'current_time')
            
            elif command == 'date' or 'дата' in message_lower:
                # Плагин даты
                return self.plugin_manager.execute_plugin('datetime', 'current_date')
            
            elif command == 'search' or 'найди' in message_lower:
                # Плагин веб-поиска
                query = parts[1] if len(parts) > 1 else message.replace('найди', '').strip()
                return self.plugin_manager.execute_plugin('web_search', query)
        
        # Проверяем, можно ли использовать плагины на основе содержания сообщения
        elif 'посчитай' in message_lower or 'сколько будет' in message_lower:
            try:
                # Извлекаем математическое выражение
                import re
                expression = re.findall(r'[0-9+\-*/().\s]+', message)
                if expression:
                    expr = ''.join(expression).strip()
                    if expr:
                        return self.plugin_manager.execute_plugin('calculator', expr)
            except:
                pass
        
        elif any(word in message_lower for word in ['время', 'час', 'сейчас']):
            return self.plugin_manager.execute_plugin('datetime', 'current_time')
        
        elif any(word in message_lower for word in ['дата', 'число', 'сегодня']):
            return self.plugin_manager.execute_plugin('datetime', 'current_date')
        
        elif any(word in message_lower for word in ['найди', 'поищи', 'google', 'поиск']):
            # Извлекаем запрос для поиска
            query = message.replace('найди', '').replace('поищи', '').replace('google', '').replace('поиск', '').strip()
            if query:
                return self.plugin_manager.execute_plugin('web_search', query)
        
        return None
    
    def generate_response(self, message, model_type="L1", use_web_search=False, use_plugins=True):
        """
        Генерация ответа с возможностью выбора модели
        
        Args:
            message: Входное сообщение
            model_type: Тип модели (L1, L2, L3)
            use_web_search: Использовать веб-поиск
            use_plugins: Использовать плагины
        """
        # Проверяем, нужно ли сменить модель
        if self.active_model is None or not hasattr(self.active_model, 'model_id') or \
           (hasattr(self.active_model, 'model_id') and model_type.upper() not in self.active_model.model_id):
            self.load_model(f"Synthesis-{model_type.upper()}")
        
        return self.chat(message, use_web_search=use_web_search, use_plugins=use_plugins)
    
    def quantize_model(self, method='dynamic'):
        """
        Квантовать текущую модель
        
        Args:
            method: Метод квантования ('dynamic', 'static', 'qat', 'mixed')
        """
        if not self.active_model:
            logger.error("Нет активной модели для квантования")
            return False
        
        try:
            logger.info(f"Квантование модели методом {method}...")
            
            if method == 'mixed':
                self.active_model = self.advanced_quantizer.mixed_precision_quantization(self.active_model)
            else:
                self.active_model = self.quantizer.apply_quantization(self.active_model, method=method)
            
            logger.info(f"✅ Модель успешно квантована методом {method}")
            print(f"✅ Модель квантована: {method}")
            
            # Проверяем степень сжатия
            original_size, quantized_size = self._get_model_sizes()
            if original_size > 0:
                compression_ratio = original_size / quantized_size if quantized_size > 0 else 1
                logger.info(f"📊 Степень сжатия: {compression_ratio:.2f}x")
                print(f"📊 Степень сжатия: {compression_ratio:.2f}x")
            
            return True
        except Exception as e:
            logger.error(f"Ошибка квантования: {e}")
            print(f"❌ Ошибка квантования: {e}")
            return False
    
    def _get_model_sizes(self):
        """Получить размеры оригинальной и квантованной модели"""
        # Для простоты возвращаем приблизительные значения
        # В реальности нужно сравнивать до и после квантования
        return 1000000, 250000  # Заглушка
    
    def get_performance_info(self):
        """Получить информацию о производительности"""
        info = {
            'device': str(self.device),
            'gpu_available': self.gpu_accelerator.is_gpu_available(),
            'model_loaded': self.active_model is not None,
            'plugins_count': len(self.plugin_manager.list_plugins()),
            'gpu_memory_info': self.gpu_accelerator.get_memory_info() if self.gpu_accelerator.is_gpu_available() else None
        }
        
        if self.active_model and hasattr(self.active_model, 'get_info'):
            info['model_info'] = self.active_model.get_info()
        
        return info
    
    def get_dialogue_stats(self):
        """Получить статистику диалогов"""
        if not self.active_model or not hasattr(self.active_model, 'dialogue_collector'):
            return {'total_dialogues': 0, 'total_exchanges': 0}
        if self.active_model.dialogue_collector:
            return self.active_model.dialogue_collector.get_statistics()
        return {'total_dialogues': 0, 'total_exchanges': 0}
    
    def real_training(self, epochs=10, batch_size=4):
        if not self.active_model:
            logger.error("Попытка обучения без активной модели")
            print("❌ Нет активной модели")
            return False
        
        try:
            if hasattr(self.active_model, 'real_train'):
                logger.info(f"🚀 Обучение на {epochs} эпох на устройстве {self.device}...")
                print(f"🚀 Обучение на {epochs} эпох...")
                success = self.active_model.real_train(epochs=epochs, batch_size=batch_size)
                
                if success:
                    logger.info("🎉 Обучение успешно завершено!")
                    print("🎉 Обучение успешно завершено!")
                    # Тестируем после обучения
                    test_texts = ["как дела", "все работает", "что нового", "хорошая работа"]
                    logger.info("🧪 Тест после обучения:")
                    print("\n🧪 Тест после обучения:")
                    for text in test_texts:
                        response = self.chat(text)
                        logger.debug(f"Тест: '{text}' -> {response}")
                        print(f"   '{text}' -> {response}")
                else:
                    logger.error("Обучение завершилось неудачей")
                return success
            else:
                logger.error("Модель не поддерживает обучение")
                print("❌ Модель не поддерживает обучение")
                return False
        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}", exc_info=True)
            print(f"❌ Ошибка: {e}")
            return False
    
    def save_model(self, path="models/synthesis_l1.pth"):
        if not self.active_model:
            logger.error("Попытка сохранения без активной модели")
            print("❌ Нет модели для сохранения")
            return False
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Сохранение модели в {path}...")
            
            # Для GPT модели сохраняем gpt_model
            if hasattr(self.active_model, 'gpt_model'):
                torch.save({
                    'gpt_model_state_dict': self.active_model.gpt_model.state_dict(),
                    'model_id': self.active_model.model_id,
                    'version': self.active_model.version,
                    'is_trained': getattr(self.active_model, 'is_trained', True),
                }, path)
                logger.info("Сохранена GPT модель")
            else:
                torch.save(self.active_model.state_dict(), path)
            
            logger.info(f"💾 Модель сохранена: {path}")
            print(f"💾 Модель сохранена: {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}", exc_info=True)
            print(f"❌ Ошибка при сохранении: {e}")
            return False
    
    def info(self):
        info_text = f"""
🧠 {self.name} AI System v{self.version}
📊 Устройство: {self.device}
⚡ GPU Ускорение: {'Доступно' if self.gpu_accelerator.is_gpu_available() else 'Недоступно'}
🔌 Плагинов загружено: {len(self.plugin_manager.list_plugins())}

Доступные модели:
• Synthesis-L1 - GPT трансформерная модель (текстовая генерация)
• Synthesis-L2 - Продвинутая GPT-3 подобная модель (улучшенная архитектура)
• Synthesis-L3 - Ещё более продвинутая модель (масштабированная архитектура)

Доступные плагины:
"""
        
        # Добавляем список плагинов
        for plugin_name in self.plugin_manager.list_plugins():
            plugin_info = self.plugin_manager.get_plugin_info(plugin_name)
            if plugin_info:
                status = "✅" if plugin_info.enabled else "❌"
                info_text += f"  {status} {plugin_info.name} ({plugin_info.plugin_type.value}) - {plugin_info.description}\n"
        
        info_text += "\nДоступные команды плагинов:\n"
        info_text += "  !calc <выражение> или 'посчитай' - Математические вычисления\n"
        info_text += "  !time или 'время' - Текущее время\n"
        info_text += "  !date или 'дата' - Текущая дата\n"
        info_text += "  !search <запрос> или 'найди' - Веб-поиск\n"
        
        if self.active_model:
            if hasattr(self.active_model, 'get_info'):
                model_info = self.active_model.get_info()
                info_text += f"\n🎯 Активная модель: {model_info['model_id']}"
                info_text += f"\n📈 Параметров: {model_info['parameters']:,}"
            else:
                info_text += f"\n🎯 Активная модель: {self.active_model.model_id if hasattr(self.active_model, 'model_id') else 'Неизвестная модель'}"
        
        # Добавляем информацию о GPU если доступно
        if self.gpu_accelerator.is_gpu_available():
            memory_info = self.gpu_accelerator.get_memory_info()
            if memory_info:
                info_text += f"\n💾 GPU Память: {memory_info['total'] / 1024**3:.2f}GB всего, {memory_info['allocated'] / 1024**3:.2f}GB использовано"
        
        return info_text