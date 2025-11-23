import torch
import torch.nn as nn
import re
import random
import logging
import os

# Импорт logger с обработкой возможных ошибок
try:
    from cerebra.logger_config import logger
except ImportError:
    logger = logging.getLogger('cerebra')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# Импорт GPT модели и токенизатора
try:
    from .gpt_model import GPTTransformer
    from .advanced_transformer import SynthesisL2, SynthesisL3
    from .tokenizer import SimpleTokenizer
    from .training import prepare_training_data, train_gpt_model
    GPT_AVAILABLE = True
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    try:
        from cerebra.models.gpt_model import GPTTransformer
        from cerebra.models.advanced_transformer import SynthesisL2, SynthesisL3
        from cerebra.models.tokenizer import SimpleTokenizer
        from cerebra.models.training import prepare_training_data, train_gpt_model
        GPT_AVAILABLE = True
        ADVANCED_MODELS_AVAILABLE = True
    except ImportError:
        try:
            from .gpt_model import GPTTransformer
            from .tokenizer import SimpleTokenizer
            from .training import prepare_training_data, train_gpt_model
            GPT_AVAILABLE = True
            ADVANCED_MODELS_AVAILABLE = False
        except ImportError:
            try:
                from cerebra.models.gpt_model import GPTTransformer
                from cerebra.models.tokenizer import SimpleTokenizer
                from cerebra.models.training import prepare_training_data, train_gpt_model
                GPT_AVAILABLE = True
                ADVANCED_MODELS_AVAILABLE = False
            except ImportError:
                GPT_AVAILABLE = False
                ADVANCED_MODELS_AVAILABLE = False
                logger.warning("GPT компоненты недоступны, используется старая LSTM модель")

# Импорт сборщика диалогов
try:
    from ..dialogue_training import dialogue_collector
except ImportError:
    try:
        from cerebra.dialogue_training import dialogue_collector
    except ImportError:
        dialogue_collector = None

class SynthesisL1(nn.Module):
    """
    Главная модель Synthesis-L1 - современная GPT трансформерная модель
    """
    def __init__(self, use_gpt=True):
        super().__init__()
        self.model_id = "Synthesis-L1"
        self.version = "2.0.0" if use_gpt and GPT_AVAILABLE else "1.0.0"
        self.use_gpt = use_gpt and GPT_AVAILABLE
        
        if self.use_gpt:
            # GPT Transformer модель
            self.tokenizer = SimpleTokenizer(vocab_size=10000)
            self.gpt_model = GPTTransformer(
                vocab_size=10000,
                d_model=512,
                n_heads=8,
                n_layers=6,
                d_ff=2048,
                max_seq_len=512,
                dropout=0.1
            )
            
            # Загружаем токенизатор если есть
            tokenizer_path = "models/tokenizer.json"
            if os.path.exists(tokenizer_path):
                try:
                    self.tokenizer.load(tokenizer_path)
                    logger.info(f"Загружен токенизатор с {self.tokenizer.get_vocab_size()} токенами")
                except:
                    logger.warning("Не удалось загрузить токенизатор, будет создан новый")
        else:
            # Старая LSTM модель (fallback)
            self.vocab = {
                '<PAD>': 0, '<UNK>': 1, 'привет': 2, 'пока': 3, 'как': 4, 
                'дела': 5, 'что': 6, 'ты': 7, 'модель': 8, 'обучение': 9,
                'тест': 10, 'система': 11, 'работа': 12, 'хорошо': 13, 'плохо': 14
            }
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            self.embedding = nn.Embedding(len(self.vocab), 128)
            self.lstm = nn.LSTM(128, 256, batch_first=True)
            self.fc = nn.Linear(256, 2)
        
        # Сборщик диалогов для автоматического обучения
        self.dialogue_collector = dialogue_collector
        
        # Флаг обученности модели
        self.is_trained = False
        if self.use_gpt:
            # Проверяем, есть ли сохраненная модель
            model_path = "models/synthesis_l1_trained.pth"
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if 'gpt_model_state_dict' in checkpoint:
                        self.gpt_model.load_state_dict(checkpoint['gpt_model_state_dict'])
                        # Проверяем флаг обученности из checkpoint
                        self.is_trained = checkpoint.get('is_trained', True)
                        logger.info(f"Загружена GPT модель (обучена: {self.is_trained})")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить модель: {e}")
                    pass
        
        model_type = "GPT Transformer" if self.use_gpt else "LSTM"
        logger.info(f"🎯 Создана {self.model_id} v{self.version} ({model_type})")
        print(f"🎯 Создана {self.model_id} v{self.version} ({model_type})")
        if self.use_gpt and not self.is_trained:
            print("⚠️  Модель не обучена! Используйте пункт 2 в меню для обучения.")
    
    def text_to_tokens(self, text, device=None):
        words = re.findall(r'\b\w+\b', text.lower())
        tokens = [self.vocab.get(word, 1) for word in words]  # 1 = <UNK>
        if len(tokens) < 10:
            tokens += [0] * (10 - len(tokens))  # дополняем до 10 токенов
        else:
            tokens = tokens[:10]
        tensor = torch.tensor(tokens).unsqueeze(0)
        # Перемещаем тензор на нужное устройство
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    
    def forward(self, x):
        """Forward pass (только для LSTM модели)"""
        if self.use_gpt:
            # Для GPT используем отдельный метод generate()
            raise NotImplementedError("Для GPT используйте метод generate() через process()")
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output
    
    def process(self, text):
        # Используем GPT для генерации только если модель обучена
        if self.use_gpt and self.is_trained:
            try:
                device = next(self.gpt_model.parameters()).device
                self.gpt_model.eval()
                
                # Проверяем размер словаря токенизатора
                vocab_size = self.tokenizer.get_vocab_size()
                if vocab_size <= len(self.tokenizer.special_tokens):
                    logger.warning("Словарь токенизатора слишком мал, используем fallback")
                    raise ValueError("Токенизатор не готов")
                
                # Генерируем ответ с обработкой ошибок
                try:
                    generated = self.gpt_model.generate(
                        self.tokenizer,
                        prompt=text,
                        max_length=50,
                        temperature=0.8,
                        top_k=40,
                        top_p=0.9
                    )
                    
                    # Извлекаем только сгенерированную часть (без промпта)
                    prompt_lower = text.lower().strip()
                    generated_lower = generated.lower().strip()
                    
                    if generated_lower.startswith(prompt_lower):
                        generated = generated[len(prompt_lower):].strip()
                    
                    # Проверяем, что генерация не состоит только из UNK или специальных токенов
                    if not generated or len(generated) < 3:
                        raise ValueError("Генерация пуста")
                    
                    # Проверяем на наличие только UNK
                    unk_count = generated.lower().count('<unk>')
                    if unk_count > len(generated) * 0.5:  # Если больше 50% UNK
                        logger.warning(f"Слишком много UNK в генерации ({unk_count}/{len(generated)})")
                        raise ValueError("Генерация содержит только UNK")
                    
                    logger.debug(f"GPT сгенерировал ответ: '{generated[:50]}...'")
                    return generated
                    
                except Exception as gen_error:
                    logger.warning(f"Ошибка при генерации GPT: {gen_error}")
                    raise  # Пробрасываем дальше для fallback
            except Exception as e:
                logger.warning(f"Ошибка при генерации GPT: {e}, используем fallback")
        elif self.use_gpt and not self.is_trained:
            # Модель не обучена, сразу используем fallback
            logger.debug("Модель не обучена, используем fallback ответы")
        
        # Fallback: базовые ответы
        responses = {
            'привет': 'Привет! Я Cerebra AI с моделью Synthesis-L1. Чем могу помочь?',
            'здравствуй': 'Здравствуй! Готова к работе.',
            'как дела': 'Всё отлично! Готова к работе. А у вас как дела?',
            'что ты умеешь': 'Я могу анализировать текст, обучаться на данных и общаться.',
            'пока': 'До свидания! Удачи!',
        }
        
        text_lower = text.lower().strip()
        for key, response in responses.items():
            if key in text_lower:
                return response
        
        # Если ничего не найдено
        if self.use_gpt:
            return "Я еще не обучен достаточно хорошо. Попробуйте обучить меня через меню (пункт 2)!"
        return f"Получил: '{text}'. Обучи меня для лучших ответов!"
    
    def prepare_data(self):
        texts = [
            "как дела", "что нового", "который час", "где ты", 
            "привет мир", "работает хорошо", "отличная погода", "все нормально",
            "как тебя зовут", "что это", "где находится", "когда придешь",
            "тест системы", "все работает", "хорошая работа", "отлично получается"
        ]
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]  # 0-вопрос, 1-утверждение
        return texts, labels
    
    def real_train(self, epochs=10, batch_size=4, lr=3e-4):
        device = next(self.parameters()).device  # Определяем устройство модели
        
        if self.use_gpt:
            # GPT обучение
            logger.info(f"🚀 Начинаю GPT обучение {self.model_id} на устройстве {device}...")
            print(f"🚀 Начинаю GPT обучение {self.model_id}...")
            print(f"   Архитектура: Transformer (GPT-like)")
            print(f"   Параметров: {sum(p.numel() for p in self.gpt_model.parameters()):,}")
            
            # Подготовка данных
            texts, vocab_size = prepare_training_data(self.tokenizer)
            logger.info(f"📊 Примеров для обучения: {len(texts)}")
            logger.info(f"📚 Размер словаря: {vocab_size}")
            print(f"📊 Примеров для обучения: {len(texts)}")
            print(f"📚 Размер словаря: {vocab_size}")
            
            # Обучение GPT
            success = train_gpt_model(
                self.gpt_model,
                self.tokenizer,
                texts,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )
            
            # Сохраняем токенизатор
            os.makedirs("models", exist_ok=True)
            self.tokenizer.save("models/tokenizer.json")
            logger.info("Токенизатор сохранен")
            
            # Помечаем модель как обученную
            if success:
                self.is_trained = True
                logger.info("Модель помечена как обученная")
            
            return success
        else:
            # Старое LSTM обучение (fallback)
            logger.info(f"🎓 Начинаю LSTM обучение {self.model_id} на устройстве {device}...")
            print(f"🎓 Начинаю обучение {self.model_id}...")
            
            texts, labels = self.prepare_data()
            logger.info(f"📊 Примеров для обучения: {len(texts)}")
            print(f"📊 Примеров для обучения: {len(texts)}")
            
            inputs = []
            targets = []
            for text, label in zip(texts, labels):
                tokens = self.text_to_tokens(text, device=device)
                inputs.append(tokens)
                targets.append(torch.tensor(label, device=device))
            
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            self.train()
            
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                
                for i in range(len(inputs)):
                    optimizer.zero_grad()
                    output = self(inputs[i])
                    loss = criterion(output, targets[i].unsqueeze(0))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    pred = torch.argmax(output)
                    if pred == targets[i]:
                        correct += 1
                
            accuracy = 100 * correct / len(inputs)
            avg_loss = total_loss / len(inputs)
            log_msg = f"   Эпоха {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%"
            logger.info(log_msg)
            print(log_msg)
        
        # Помечаем модель как обученную
        if self.use_gpt:
            self.is_trained = True
            logger.info("Модель помечена как обученная")
        
        logger.info("✅ Обучение завершено!")
        print("✅ Обучение завершено!")
        return True
    
    def get_info(self):
        if self.use_gpt:
            info = self.gpt_model.get_info()
            info['tokenizer_size'] = self.tokenizer.get_vocab_size()
            return info
        else:
            return {
                'model_id': self.model_id,
                'version': self.version,
                'parameters': sum(p.numel() for p in self.parameters()),
                'architecture': 'LSTM'
            }