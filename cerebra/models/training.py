"""
Обучение GPT трансформера на реальных данных
Файл: training.py - Модуль обучения модели
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os

try:
    from cerebra.logger_config import logger
except ImportError:
    import logging
    logger = logging.getLogger('cerebra')


class TextDataset(Dataset):
    """Датасет для обучения GPT"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Кодируем текст
        tokens = self.tokenizer.encode(text)
        
        # Обрезаем до max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Создаем input и target (сдвиг на 1 токен для next token prediction)
        input_ids = tokens[:-1] if len(tokens) > 1 else tokens
        target_ids = tokens[1:] if len(tokens) > 1 else tokens
        
        # Padding до max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.special_tokens['<PAD>']] * pad_len
            target_ids = target_ids + [self.tokenizer.special_tokens['<PAD>']] * pad_len
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def prepare_training_data(tokenizer, data_sources=None):
    """
    Подготовка данных для обучения
    
    data_sources: список источников данных или None для использования встроенных данных
    """
    texts = []
    
    # Встроенные данные для обучения (русский язык)
    builtin_texts = [
        "Привет! Как дела?",
        "Все отлично, спасибо!",
        "Что ты умеешь?",
        "Я умею общаться и отвечать на вопросы.",
        "Расскажи о себе.",
        "Я - Cerebra AI, языковая модель на основе трансформера.",
        "Как тебя можно обучать?",
        "Можно добавлять тексты для обучения через меню.",
        "Понятно, спасибо!",
        "Пожалуйста! Всегда рад помочь.",
        "Какая сегодня погода?",
        "К сожалению, я не могу проверять погоду в реальном времени.",
        "Хорошо, понял.",
        "Если есть еще вопросы - задавай!",
        "Что такое искусственный интеллект?",
        "Искусственный интеллект - это область компьютерных наук, которая создает системы, способные выполнять задачи, требующие человеческого интеллекта.",
        "Интересно, расскажи больше.",
        "ИИ используется в распознавании речи, обработке изображений, машинном переводе и многих других областях.",
        "Спасибо за информацию!",
        "Пожалуйста! Рад был помочь.",
        "До свидания!",
        "До встречи! Хорошего дня!",
    ]
    
    if data_sources:
        texts.extend(data_sources)
    else:
        texts.extend(builtin_texts)
    
    # Загружаем диалоги из реальных разговоров
    try:
        from cerebra.dialogue_training import dialogue_collector
        dialogue_texts = dialogue_collector.get_training_texts()
        texts.extend(dialogue_texts)
        logger.info(f"Добавлено {len(dialogue_texts)} текстов из реальных диалогов")
    except Exception as e:
        logger.warning(f"Не удалось загрузить диалоги: {e}")
        pass
    
    # Построение словаря
    vocab_size = tokenizer.build_vocab(texts)
    logger.info(f"Построен словарь размером {vocab_size}")
    
    return texts, vocab_size


def train_gpt_model(model, tokenizer, texts, epochs=10, batch_size=4, lr=3e-4, device='cpu'):
    """
    Обучение GPT модели
    
    model: GPTTransformer модель
    tokenizer: SimpleTokenizer
    texts: список текстов для обучения
    epochs: количество эпох
    batch_size: размер батча
    lr: learning rate
    device: устройство (cpu/cuda)
    """
    model = model.to(device)
    
    # Создаем датасет
    dataset = TextDataset(texts, tokenizer, max_length=model.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Оптимизатор и функция потерь
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens['<PAD>'])
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_tokens = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids)  # [batch_size, seq_len, vocab_size]
            
            # Reshape для loss
            logits_flat = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, vocab_size]
            targets_flat = target_ids.view(-1)  # [batch_size * seq_len]
            
            # Loss
            loss = criterion(logits_flat, targets_flat)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Статистика
            batch_loss = loss.item()
            batch_tokens = (targets_flat != tokenizer.special_tokens['<PAD>']).sum().item()
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / max(total_tokens, 1)
                logger.debug(f"Эпоха {epoch+1}/{epochs}, Батч {batch_idx+1}, Loss: {avg_loss:.4f}")
        
        # Эпоховая статистика
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        log_msg = f"Эпоха {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}"
        logger.info(log_msg)
        print(f"   {log_msg}")
    
    model.eval()
    logger.info("✅ Обучение GPT завершено!")
    return True

