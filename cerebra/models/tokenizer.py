"""
Токенизатор для русского языка (BPE-like)
Файл: tokenizer.py - Преобразование текста в токены и обратно, построение словаря
"""
import re
import json
import os
from collections import Counter, defaultdict

class SimpleTokenizer:
    """Простой токенизатор с поддержкой BPE"""
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Beginning of sequence
            '<EOS>': 3,  # End of sequence
            '<SEP>': 4,  # Separator
        }
        self.eos_token_id = self.special_tokens['<EOS>']
        self.unk_token_id = self.special_tokens['<UNK>']
        self._build_vocab_from_special()
    
    def _build_vocab_from_special(self):
        """Инициализация словаря специальными токенами"""
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)
    
    def _tokenize_text(self, text):
        """Разбиение текста на токены (слова + подстроки)"""
        # Нормализация
        text = text.lower().strip()
        # Разбиение на слова и знаки препинания
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens
    
    def build_vocab(self, texts):
        """Построение словаря из текстов"""
        word_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize_text(text)
            word_counts.update(tokens)
        
        # Добавляем самые частые слова в словарь
        for word, count in word_counts.most_common(self.vocab_size - len(self.word_to_id)):
            if word not in self.word_to_id:
                self.word_to_id[word] = self.next_id
                self.id_to_word[self.next_id] = word
                self.next_id += 1
        
        return len(self.word_to_id)
    
    def encode(self, text):
        """Кодирование текста в последовательность токенов"""
        tokens = self._tokenize_text(text)
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.unk_token_id)
        return ids
    
    def decode(self, ids):
        """Декодирование последовательности токенов в текст"""
        words = []
        for id_val in ids:
            if id_val in self.id_to_word:
                word = self.id_to_word[id_val]
                if word not in self.special_tokens:
                    words.append(word)
            elif id_val != self.unk_token_id:
                words.append('<UNK>')
        
        # Собираем текст, добавляя пробелы между словами
        text = ' '.join(words)
        # Убираем лишние пробелы вокруг знаков препинания
        text = re.sub(r'\s+([^\w\s])', r'\1', text)
        text = re.sub(r'([^\w\s])\s+', r'\1 ', text)
        return text.strip()
    
    def save(self, path):
        """Сохранение токенизатора"""
        data = {
            'vocab_size': self.vocab_size,
            'word_to_id': self.word_to_id,
            'id_to_word': {int(k): v for k, v in self.id_to_word.items()},
            'special_tokens': self.special_tokens
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path):
        """Загрузка токенизатора"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.word_to_id = data['word_to_id']
        self.id_to_word = {int(k): v for k, v in data['id_to_word'].items()}
        self.special_tokens = data['special_tokens']
        self.eos_token_id = self.special_tokens['<EOS>']
        self.unk_token_id = self.special_tokens['<UNK>']
        self.next_id = len(self.word_to_id)
    
    def get_vocab_size(self):
        """Получить размер словаря"""
        return len(self.word_to_id)

