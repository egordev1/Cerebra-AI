#!/usr/bin/env python3
"""
Простой запуск Cerebra AI
"""

from cerebra import ai

def main():
    print("🚀 Запуск Cerebra AI...")
    
    # Загрузка модели
    model = ai.load_model("Synthesis-L1")
    
    # Информация
    print(ai.info())
    
    # Чат
    print("\n💬 Тестируем чат:")
    while True:
        user_input = input("\n👤 Вы: ")
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break
        response = ai.chat(user_input)
        print(f"🤖 Cerebra: {response}")

if __name__ == "__main__":
    main()