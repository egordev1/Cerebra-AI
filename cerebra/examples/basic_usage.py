#!/usr/bin/env python3
"""
Пример использования Cerebra AI
"""

import sys
import os

# Автоматическое определение путей
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Добавляем корень проекта в Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from cerebra import ai
    from cerebra.utils import print_system_info
    print("✅ Модули успешно импортированы!")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Python path: {sys.path}")
    sys.exit(1)

def main():
    # Информация о системе
    print_system_info()
    
    # Загрузка модели
    print("\n1. Загрузка модели...")
    model = ai.load_model("Synthesis-L1")
    
    # Информация о системе
    print("\n2. Информация о системе:")
    print(ai.info())
    
    # Тест диалога
    print("\n3. Тест диалога:")
    test_messages = [
        "привет Cerebra",
        "расскажи о себе",
        "что такое Synthesis-L1?",
        "как тебя обучать?",
        "пока"
    ]
    
    for message in test_messages:
        response = ai.chat(message)
        print(f"👤: {message}")
        print(f"🤖: {response}\n")
    
    # Демонстрация обучения
    print("\n4. Демонстрация обучения:")
    ai.train_model(epochs=2)
    
    # Сохранение модели
    print("\n5. Сохранение модели...")
    ai.save_model("models/synthesis_l1_demo.pth")
    
    print("\n🎉 Cerebra AI готова к работе!")

if __name__ == "__main__":
    main()