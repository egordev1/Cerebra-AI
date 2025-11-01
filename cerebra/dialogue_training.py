"""
Автоматическое обучение на реальных диалогах из чата
Файл: dialogue_training.py - Сборщик диалогов для автоматического обучения модели
"""
import os
import json
import logging
from datetime import datetime
from collections import defaultdict

try:
    from cerebra.logger_config import logger
except ImportError:
    logger = logging.getLogger('cerebra')


class DialogueCollector:
    """Сборщик диалогов из реальных разговоров"""
    
    def __init__(self, data_file="training_data/dialogues.json"):
        self.data_file = data_file
        self.dialogues = []
        self.conversation_buffer = []  # Текущий диалог
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        self.load_dialogues()
    
    def add_exchange(self, user_message: str, bot_response: str):
        """Добавить обмен сообщениями в текущий диалог"""
        self.conversation_buffer.append({
            'user': user_message.strip(),
            'bot': bot_response.strip(),
            'timestamp': datetime.now().isoformat()
        })
        
        # Сохраняем каждые 10 обменов
        if len(self.conversation_buffer) >= 10:
            self.save_conversation()
    
    def save_conversation(self):
        """Сохранить текущий диалог"""
        if self.conversation_buffer:
            self.dialogues.append({
                'messages': self.conversation_buffer.copy(),
                'created_at': datetime.now().isoformat()
            })
            self.conversation_buffer = []
            self.save_dialogues()
    
    def save_dialogues(self):
        """Сохранить все диалоги в файл"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.dialogues, f, ensure_ascii=False, indent=2)
            logger.debug(f"Сохранено {len(self.dialogues)} диалогов")
        except Exception as e:
            logger.error(f"Ошибка при сохранении диалогов: {e}")
    
    def load_dialogues(self):
        """Загрузить сохраненные диалоги"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.dialogues = json.load(f)
                logger.info(f"Загружено {len(self.dialogues)} диалогов из файла")
            except Exception as e:
                logger.warning(f"Ошибка при загрузке диалогов: {e}")
                self.dialogues = []
    
    def get_training_texts(self) -> list:
        """
        Получить тексты для обучения из диалогов
        
        Returns:
            Список текстов (вопросы и ответы)
        """
        texts = []
        
        for dialogue in self.dialogues:
            for msg in dialogue.get('messages', []):
                user_msg = msg.get('user', '').strip()
                bot_msg = msg.get('bot', '').strip()
                
                if user_msg and bot_msg:
                    # Форматируем как диалог
                    texts.append(f"Пользователь: {user_msg}")
                    texts.append(f"Ассистент: {bot_msg}")
        
        logger.info(f"Подготовлено {len(texts)} текстов для обучения из диалогов")
        return texts
    
    def get_statistics(self) -> dict:
        """Получить статистику по диалогам"""
        total_exchanges = sum(len(d.get('messages', [])) for d in self.dialogues)
        return {
            'total_dialogues': len(self.dialogues),
            'total_exchanges': total_exchanges,
            'current_buffer_size': len(self.conversation_buffer)
        }


# Глобальный экземпляр
dialogue_collector = DialogueCollector()

