#!/usr/bin/env python3
"""
Пример использования Cerebra AI
Обновленный пример с поддержкой всех моделей (L1, L2, L3)
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
    
    # Загрузка и тестирование разных моделей
    models_to_test = ["Synthesis-L1", "Synthesis-L2", "Synthesis-L3"]
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧪 Тестирование модели: {model_name}")
        print(f"{'='*60}")
        
        # Загрузка модели
        print(f"1. Загрузка модели {model_name}...")
        model = ai.load_model(model_name)
        
        if model:
            # Информация о модели
            print(f"\n2. Информация о модели:")
            print(ai.info())
            
            # Тест диалога
            print(f"\n3. Тест диалога с {model_name}:")
            test_messages = [
                "привет Cerebra",
                "расскажи о себе",
                f"что ты можешь как {model_name}?",
                "как работает трансформер?",
                "пока"
            ]
            
            for message in test_messages:
                response = ai.chat(message)
                print(f"👤: {message}")
                print(f"🤖: {response}\n")
            
            # Сохранение модели
            print(f"4. Сохранение модели {model_name}...")
            ai.save_model(f"models/synthesis_{model_name.split('-')[1].lower()}_demo.pth")
        else:
            print(f"❌ Не удалось загрузить модель {model_name}")
    
    # Демонстрация смены моделей
    print(f"\n{'='*60}")
    print("🔄 Демонстрация смены моделей")
    print(f"{'='*60}")
    
    # Переключение на L2
    ai.load_model("Synthesis-L2")
    print("\n💬 Переключение на L2:")
    response = ai.chat("Теперь ты используешь более продвинутую архитектуру L2?")
    print(f"🤖: {response}")
    
    # Переключение на L3
    ai.load_model("Synthesis-L3")
    print("\n💬 Переключение на L3:")
    response = ai.chat("Теперь ты используешь масштабированную архитектуру L3?")
    print(f"🤖: {response}")
    
    # Возврат к L1
    ai.load_model("Synthesis-L1")
    
    print(f"\n🎉 Cerebra AI с поддержкой всех моделей готова к работе!")

if __name__ == "__main__":
    main()