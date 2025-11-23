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

class Cerebra:
    def __init__(self):
        self.name = "Cerebra"
        self.version = "1.0.0"
        self.active_model = None
        
        # Определение устройства с логированием
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
            logger.info(f"📊 CUDA версия: {torch.version.cuda}")
        else:
            self.device = torch.device('cpu')
            logger.warning("⚠️ CUDA недоступна, используется CPU")
        
        logger.info(f"🧠 Запущена {self.name} AI System")
        logger.info(f"📊 Устройство: {self.device}")
        print(f"🧠 Запущена {self.name} AI System")
        print(f"📊 Устройство: {self.device}")
    
    def load_model(self, model_name="Synthesis-L1"):
        try:
            if model_name == "Synthesis-L1":
                from .models.main_model import SynthesisL1
                logger.info(f"Загрузка модели {model_name} на устройство {self.device}...")
                self.active_model = SynthesisL1(use_gpt=True)
                
                # Перемещаем модель на устройство
                if hasattr(self.active_model, 'gpt_model'):
                    self.active_model.gpt_model = self.active_model.gpt_model.to(self.device)
                self.active_model = self.active_model.to(self.device)
                logger.info(f"✅ Загружена модель: {model_name} на {self.device}")
                print(f"✅ Загружена модель: {model_name}")
            elif model_name == "Synthesis-L2":
                from .models.advanced_transformer import SynthesisL2
                logger.info(f"Загрузка модели {model_name} на устройство {self.device}...")
                self.active_model = SynthesisL2()
                self.active_model = self.active_model.to(self.device)
                logger.info(f"✅ Загружена модель: {model_name} на {self.device}")
                print(f"✅ Загружена модель: {model_name}")
            elif model_name == "Synthesis-L3":
                from .models.advanced_transformer import SynthesisL3
                logger.info(f"Загрузка модели {model_name} на устройство {self.device}...")
                self.active_model = SynthesisL3()
                self.active_model = self.active_model.to(self.device)
                logger.info(f"✅ Загружена модель: {model_name} на {self.device}")
                print(f"✅ Загружена модель: {model_name}")
            else:
                logger.error(f"Модель {model_name} не найдена")
                print(f"❌ Модель {model_name} не найдена")
                return None
            return self.active_model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}", exc_info=True)
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
                        self.active_model = SynthesisL2().to(self.device)
                    elif model_name == "Synthesis-L3":
                        from .models.advanced_transformer import SynthesisL3
                        self.active_model = SynthesisL3().to(self.device)
                    logger.info(f"✅ Модель загружена на CPU")
                    print(f"✅ Загружена модель: {model_name} на CPU (fallback)")
                    return self.active_model
                except Exception as e2:
                    logger.error(f"Ошибка при загрузке на CPU: {e2}", exc_info=True)
            return None
    
    def chat(self, message, use_web_search=False):
        """
        Общение с ИИ
        
        Args:
            message: Сообщение пользователя
            use_web_search: Использовать веб-поиск для ответа
        """
        if not self.active_model:
            logger.warning("Попытка чата без загруженной модели")
            return "⚠️ Сначала загрузите модель: ai.load_model()"
        
        logger.debug(f"Обработка сообщения: {message}")
        
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
                    return web_answer
            except Exception as e:
                logger.warning(f"Ошибка веб-поиска: {e}")
        
        # Генерация ответа моделью
        response = self.active_model.process(message)
        logger.debug(f"Получен ответ: {response}")
        
        # Сохраняем диалог для автоматического обучения
        if hasattr(self.active_model, 'dialogue_collector') and self.active_model.dialogue_collector:
            self.active_model.dialogue_collector.add_exchange(message, response)
        
        return response
    
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

Доступные модели:
• Synthesis-L1 - GPT трансформерная модель (текстовая генерация)
• Synthesis-L2 - Продвинутая GPT-3 подобная модель (улучшенная архитектура)
• Synthesis-L3 - Ещё более продвинутая модель (масштабированная архитектура)
"""
        if self.active_model:
            if hasattr(self.active_model, 'get_info'):
                model_info = self.active_model.get_info()
                info_text += f"\n🎯 Активная модель: {model_info['model_id']}"
                info_text += f"\n📈 Параметров: {model_info['parameters']:,}"
            else:
                info_text += f"\n🎯 Активная модель: {self.active_model.model_id if hasattr(self.active_model, 'model_id') else 'Неизвестная модель'}"
        
        return info_text