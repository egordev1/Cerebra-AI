#!/usr/bin/env python3
"""
Скрипт для очистки старых данных и сброса до заводских настроек
"""
import os
import shutil

def cleanup():
    """Очистка всех пользовательских данных"""
    print("🧹 ОЧИСТКА ПРОЕКТА ДО ЗАВОДСКИХ НАСТРОЕК")
    print("="*60)
    
    # Что будем удалять
    cleanup_items = [
        ("training_data/", "Пользовательские данные обучения"),
        ("models/*.pth", "Сохраненные модели"),
        ("models/tokenizer.json", "Токенизатор"),
        ("logs/", "Логи"),
        ("__pycache__/", "Кэш Python"),
        ("*.pyc", "Скомпилированные файлы"),
    ]
    
    print("\n📋 Будет удалено:")
    for path, desc in cleanup_items:
        print(f"   - {desc} ({path})")
    
    confirm = input("\n⚠️  Вы уверены? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Отменено")
        return
    
    removed_count = 0
    
    # Удаляем директории
    dirs_to_remove = ['training_data', 'logs', '__pycache__']
    for dir_name in dirs_to_remove:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"✅ Удалена директория: {dir_name}/")
                removed_count += 1
            except Exception as e:
                print(f"❌ Ошибка при удалении {dir_name}/: {e}")
    
    # Удаляем файлы моделей
    models_dir = 'models'
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(('.pth', '.json')):
                try:
                    os.remove(os.path.join(models_dir, file))
                    print(f"✅ Удален файл: models/{file}")
                    removed_count += 1
                except Exception as e:
                    print(f"❌ Ошибка при удалении models/{file}: {e}")
    
    # Удаляем .pyc файлы
    for root, dirs, files in os.walk('.'):
        # Пропускаем виртуальные окружения
        if 'venv' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.pyc'):
                try:
                    os.remove(os.path.join(root, file))
                    removed_count += 1
                except:
                    pass
    
    print(f"\n✅ Очистка завершена! Удалено элементов: {removed_count}")
    print("💡 Проект сброшен до заводских настроек")

if __name__ == "__main__":
    cleanup()

