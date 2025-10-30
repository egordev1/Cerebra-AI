#!/usr/bin/env python3
"""
Cerebra AI - Главный запускаемый файл
"""

import sys
import os
import time

# Добавляем текущую папку в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cerebra import ai
from cerebra.utils import print_system_info

def show_menu():
    print("\n" + "="*50)
    print("🧠 CEREBRA AI - ГЛАВНОЕ МЕНЮ")
    print("="*50)
    print("1. 💬 Чат с ИИ")
    print("2. 🎓 Обучить модель")
    print("3. ℹ️  Информация о системе")
    print("4. 💾 Сохранить модель")
    print("5. 🚪 Выход")
    print("="*50)

def chat_mode():
    print("\n💬 РЕЖИМ ЧАТА (для выхода введите 'выход')")
    while True:
        user_input = input("\n👤 Вы: ").strip()
        if user_input.lower() in ['выход', 'exit', 'quit']:
            break
        response = ai.chat(user_input)
        print(f"🤖 Cerebra: {response}")

def training_mode():
    print("\n🎓 РЕЖИМ ОБУЧЕНИЯ")
    try:
        epochs = int(input("Введите количество эпох (рекомендуется 3-10): ") or "5")
        success = ai.real_training(epochs=epochs)
        
        if success:
            print("\n✅ Обучение завершено!")
            save = input("Сохранить модель? (y/n): ").lower()
            if save == 'y':
                ai.save_model("models/synthesis_l1_trained.pth")
        else:
            print("❌ Обучение не удалось")
    except ValueError:
        print("❌ Введите число!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def main():
    print("🚀 ЗАГРУЗКА CERBERA AI...")
    time.sleep(1)
    
    # Информация о системе
    print_system_info()
    time.sleep(1)
    
    # Загрузка модели
    print("\n📦 Загрузка модели Synthesis-L1...")
    ai.load_model("Synthesis-L1")
    print("✅ Модель загружена!")
    
    # Главный цикл
    while True:
        show_menu()
        choice = input("\nВыберите действие (1-5): ").strip()
        
        if choice == '1':
            chat_mode()
        elif choice == '2':
            training_mode()
        elif choice == '3':
            print(ai.info())
        elif choice == '4':
            ai.save_model("models/synthesis_l1.pth")
        elif choice == '5':
            print("\n👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор!")
        
        input("\nНажмите Enter чтобы продолжить...")

if __name__ == "__main__":
    main()