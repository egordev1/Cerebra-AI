#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cerebra AI - Главный запускаемый файл приложения
Файл: main.py - Точка входа в приложение, меню, интерфейс пользователя
Современная AI система на основе GPT трансформера
"""

import sys
import os
import time
import io

# Установка UTF-8 кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Добавляем текущую папку в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cerebra import ai
from cerebra.utils import print_system_info
from cerebra.logger_config import logger


def show_menu():
    """Отображение главного меню"""
    stats = ai.get_dialogue_stats()
    print("\n" + "="*60)
    print("🧠 CEREBRA AI - ГЛАВНОЕ МЕНЮ")
    print("🤖 Модель: Synthesis-L1")
    print("="*60)
    print("1. 💬 Чат с ИИ (с веб-поиском)")
    print("2. 🎓 Обучение модели (на реальных диалогах)")
    print("3. 🌐 Поиск в интернете")
    print("4. ℹ️  Информация о системе")
    print("5. 💾 Сохранить модель")
    print("6. 🗑️  Очистить историю диалогов")
    print("7. 🚪 Выход")
    if stats['total_dialogues'] > 0:
        print(f"\n📊 Статистика: {stats['total_dialogues']} диалогов, {stats['total_exchanges']} обменов")
    print("="*60)


def chat_mode():
    """Режим чата с опциональным веб-поиском"""
    logger.info("💬 РЕЖИМ ЧАТА")
    print("\n💬 РЕЖИМ ЧАТА")
    print("💡 Подсказка: Модель автоматически сохраняет диалоги для обучения")
    print("   Для выхода введите 'выход'\n")
    
    use_web = input("Использовать веб-поиск? (y/n, по умолчанию n): ").strip().lower() == 'y'
    
    try:
        while True:
            user_input = input("\n👤 Вы: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['выход', 'exit', 'quit']:
                break
            
            logger.debug(f"Пользовательский ввод: {user_input}")
            
            # Автоматически используем веб-поиск для определенных запросов
            auto_web_keywords = ['что такое', 'кто такой', 'когда', 'где', 'как найти', 
                                'новости', 'погода', 'курс валют', 'расписание']
            should_search = use_web or any(kw in user_input.lower() for kw in auto_web_keywords)
            
            if should_search:
                print("🔍 Ищу информацию в интернете...")
            
            response = ai.chat(user_input, use_web_search=should_search)
            print(f"🤖 Cerebra: {response}")
            logger.debug(f"Ответ модели: {response}")
            
    except (KeyboardInterrupt, EOFError):
        logger.info("Выход из режима чата")
        print("\n\n👋 Выход из режима чата")


def training_mode():
    """Режим обучения на реальных диалогах"""
    logger.info("🎓 РЕЖИМ ОБУЧЕНИЯ")
    print("\n🎓 ОБУЧЕНИЕ МОДЕЛИ НА РЕАЛЬНЫХ ДИАЛОГАХ")
    print("="*60)
    
    stats = ai.get_dialogue_stats()
    print(f"📊 Доступно диалогов: {stats['total_dialogues']}")
    print(f"📊 Всего обменов: {stats['total_exchanges']}")
    
    if stats['total_exchanges'] < 10:
        print("\n⚠️  Недостаточно данных для обучения!")
        print("   Рекомендуется накопить хотя бы 10-20 обменов в чате.")
        print("   Продолжить обучение? (y/n): ", end='')
        choice = input().strip().lower()
        if choice != 'y':
            return
    
    try:
        epochs_input = input("\nВведите количество эпох (рекомендуется 5-20): ").strip() or "10"
        epochs = int(epochs_input)
        
        batch_size_input = input("Размер батча (рекомендуется 2-4): ").strip() or "4"
        batch_size = int(batch_size_input)
        
        logger.info(f"Начато обучение: {epochs} эпох, batch_size={batch_size}")
        print(f"\n🚀 Запуск обучения...")
        print(f"   Эпох: {epochs}")
        print(f"   Размер батча: {batch_size}")
        print(f"   Данные: реальные диалоги\n")
        
        success = ai.real_training(epochs=epochs, batch_size=batch_size)
        
        if success:
            logger.info("Обучение успешно завершено")
            print("\n✅ Обучение завершено!")
            save = input("\nСохранить модель? (y/n): ").strip().lower()
            if save == 'y':
                logger.info("Сохранение модели...")
                ai.save_model("models/synthesis_l1_trained.pth")
                print("✅ Модель сохранена!")
        else:
            logger.error("Обучение не удалось")
            print("❌ Обучение не удалось")
            
    except ValueError:
        logger.error("Ошибка: введено не число")
        print("❌ Введите число!")
    except (KeyboardInterrupt, EOFError):
        logger.info("Обучение прервано пользователем")
        print("\n\n👋 Обучение прервано")
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}", exc_info=True)
        print(f"❌ Ошибка: {e}")


def web_search_mode():
    """Режим поиска в интернете"""
    logger.info("🌐 РЕЖИМ ПОИСКА В ИНТЕРНЕТЕ")
    print("\n🌐 ПОИСК В ИНТЕРНЕТЕ")
    print("Введите поисковый запрос (или 'выход' для выхода)\n")
    
    try:
        from cerebra.web_search import web_searcher
        
        while True:
            query = input("🔍 Поиск: ").strip()
            
            if not query or query.lower() in ['выход', 'exit', 'quit']:
                break
            
            print("\n🔍 Ищу...")
            results = web_searcher.search(query, max_results=5)
            
            if results:
                print(f"\n📊 Найдено результатов: {len(results)}\n")
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result.get('title', 'Без названия')}")
                    if result.get('snippet'):
                        print(f"   {result['snippet']}")
                    if result.get('url'):
                        print(f"   🔗 {result['url']}")
                    print()
            else:
                print("❌ Результаты не найдены")
            
            print("-"*60)
            
    except ImportError:
        print("❌ Модуль веб-поиска недоступен")
    except (KeyboardInterrupt, EOFError):
        logger.info("Поиск прерван")
        print("\n\n👋 Поиск завершен")


def clear_dialogues():
    """Очистка истории диалогов"""
    logger.info("🗑️ ОЧИСТКА ДИАЛОГОВ")
    print("\n🗑️ ОЧИСТКА ИСТОРИИ ДИАЛОГОВ")
    
    stats = ai.get_dialogue_stats()
    print(f"📊 Текущая статистика:")
    print(f"   Диалогов: {stats['total_dialogues']}")
    print(f"   Обменов: {stats['total_exchanges']}")
    
    confirm = input("\n⚠️  Вы уверены? Это действие нельзя отменить! (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        try:
            from cerebra.dialogue_training import dialogue_collector
            dialogue_collector.dialogues = []
            dialogue_collector.conversation_buffer = []
            dialogue_collector.save_dialogues()
            
            # Удаляем файл
            if os.path.exists(dialogue_collector.data_file):
                os.remove(dialogue_collector.data_file)
            
            logger.info("История диалогов очищена")
            print("✅ История диалогов очищена!")
        except Exception as e:
            logger.error(f"Ошибка при очистке: {e}")
            print(f"❌ Ошибка: {e}")
    else:
        print("❌ Отменено")


def main():
    """Главная функция"""
    logger.info("🚀 ЗАГРУЗКА CEREBRA AI...")
    print("🚀 ЗАГРУЗКА CEREBRA AI...")
    time.sleep(0.5)
    
    # Информация о системе
    print_system_info()
    time.sleep(0.5)
    
    # Загрузка модели
    logger.info("Загрузка модели Synthesis-L1...")
    print("\n📦 Загрузка модели Synthesis-L1...")
    model = ai.load_model("Synthesis-L1")
    
    if model:
        logger.info("Модель успешно загружена")
        print("✅ Модель загружена!\n")
    else:
        logger.error("Не удалось загрузить модель")
        print("❌ Не удалось загрузить модель!")
        return
    
    # Главный цикл
    try:
        while True:
            show_menu()
            choice = input("\nВыберите действие (1-7): ").strip()
            
            if choice == '1':
                chat_mode()
            elif choice == '2':
                training_mode()
            elif choice == '3':
                web_search_mode()
            elif choice == '4':
                info = ai.info()
                logger.info("Запрос информации о системе")
                print(info)
            elif choice == '5':
                logger.info("Сохранение модели...")
                print("\n💾 Сохранение модели...")
                success = ai.save_model("models/synthesis_l1.pth")
                if success:
                    print("✅ Модель сохранена!")
            elif choice == '6':
                clear_dialogues()
            elif choice == '7':
                logger.info("Выход из приложения")
                print("\n👋 До свидания!")
                break
            else:
                logger.warning(f"Неверный выбор пользователя: {choice}")
                print("❌ Неверный выбор!")
            
            if choice != '7':
                try:
                    input("\nНажмите Enter чтобы продолжить...").strip()
                except (KeyboardInterrupt, EOFError):
                    logger.info("Выход из приложения (прервано пользователем)")
                    print("\n\n👋 До свидания!")
                    break
                    
    except (KeyboardInterrupt, EOFError):
        logger.info("Выход из приложения (прервано пользователем)")
        print("\n\n👋 До свидания!")


if __name__ == "__main__":
    main()
