#!/usr/bin/env python3
"""
Запуск Cerebra AI - выбор интерфейса
"""

import sys
import os

def main():
    print("🚀 Запуск Cerebra AI...")
    print("\nВыберите режим работы:")
    print("1. 🖥️  Консольный интерфейс")
    print("2. 🖥️  Графический интерфейс (GUI)")
    print("3. 🏗️  Расширенный консольный интерфейс")
    
    choice = input("\nВведите номер (1-3) или 'q' для выхода: ").strip()
    
    if choice == '1':
        # Простой консольный режим
        from cerebra import ai
        
        print("\n🚀 Запуск Cerebra AI (Консольный режим)...")
        
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
    
    elif choice == '2':
        # Графический интерфейс
        try:
            import tkinter as tk
            from gui_cerebra import main as gui_main
            gui_main()
        except ImportError as e:
            print(f"❌ Не удалось запустить GUI: {e}")
            print("💡 Установите необходимые зависимости: pip install tkinter")
            sys.exit(1)
    
    elif choice == '3':
        # Расширенный консольный интерфейс
        os.system("python build_cerebra.py")
    
    elif choice.lower() in ['q', 'quit', 'exit']:
        print("\n👋 До свидания!")
        sys.exit(0)
    
    else:
        print("\n❌ Неверный выбор!")
        main()

if __name__ == "__main__":
    main()