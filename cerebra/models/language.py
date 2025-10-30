import torch
import torch.nn as nn
import re
import random

class SynthesisL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_id = "Synthesis-L1"
        self.version = "1.0.0"
        
        # Словарь
        self.vocab = {
            '<PAD>': 0, '<UNK>': 1, 'привет': 2, 'пока': 3, 'как': 4, 
            'дела': 5, 'что': 6, 'ты': 7, 'модель': 8, 'обучение': 9,
            'тест': 10, 'система': 11, 'работа': 12, 'хорошо': 13, 'плохо': 14
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Архитектура нейросети
        self.embedding = nn.Embedding(len(self.vocab), 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, 2)  # 2 класса: вопрос или утверждение
        
        print(f"🎯 Создана {self.model_id}")
    
    def text_to_tokens(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        tokens = [self.vocab.get(word, 1) for word in words]  # 1 = <UNK>
        if len(tokens) < 10:
            tokens += [0] * (10 - len(tokens))  # дополняем до 10 токенов
        else:
            tokens = tokens[:10]
        return torch.tensor(tokens).unsqueeze(0)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output
    
    def process(self, text):
        responses = {
            'привет': 'Привет! Я Cerebra AI с моделью Synthesis-L1.',
            'как дела': 'Всё отлично! Готова к работе.',
            'что ты умеешь': 'Я могу анализировать текст и обучаться.',
            'обучение': 'Используйте real_training() для обучения.',
            'пока': 'До свидания!'
        }
        
        text_lower = text.lower()
        for key, response in responses.items():
            if key in text_lower:
                return response
        
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
    
    def real_train(self, epochs=5):
        print(f"🎓 Начинаю обучение {self.model_id}...")
        
        texts, labels = self.prepare_data()
        print(f"📊 Примеров для обучения: {len(texts)}")
        
        # Подготовка данных
        inputs = []
        targets = []
        for text, label in zip(texts, labels):
            tokens = self.text_to_tokens(text)
            inputs.append(tokens)
            targets.append(torch.tensor(label))
        
        # Обучение
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
                
                # Считаем точность
                pred = torch.argmax(output)
                if pred == targets[i]:
                    correct += 1
            
            accuracy = 100 * correct / len(inputs)
            avg_loss = total_loss / len(inputs)
            print(f"   Эпоха {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")
        
        print("✅ Обучение завершено!")
        return True
    
    def get_info(self):
        return {
            'model_id': self.model_id,
            'version': self.version,
            'parameters': sum(p.numel() for p in self.parameters())
        }