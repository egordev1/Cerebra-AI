import sys
import os
import time

sys.path.append('.')

from cerebra import ai
from cerebra.utils import print_system_info

def main():
    print("🎓 ОБУЧЕНИЕ CERBERA AI")
    print("=" * 40)
    
    # Информация о системе
    print_system_info()
    
    # Загрузка модели
    print("\n1. Загрузка модели...")
    model = ai.load_model("Synthesis-L1")
    
    # Информация
    print(ai.info())
    
    # Тест до обучения
    print("\n2. Тест ДО обучения:")
    test_messages = ["привет", "как дела", "что ты умеешь", "тест"]
    for msg in test_messages:
        print(f"👤: {msg}")
        print(f"🤖: {ai.chat(msg)}")
    
    # Обучение
    print("\n3. 🚀 ЗАПУСК ОБУЧЕНИЯ...")
    start_time = time.time()
    
    success = ai.real_training(epochs=5)
    
    end_time = time.time()
    
    if success:
        print(f"✅ Обучение заняло: {end_time - start_time:.1f} сек")
        
        # Тест после обучения
        print("\n4. Тест ПОСЛЕ обучения:")
        for msg in test_messages:
            print(f"👤: {msg}")
            print(f"🤖: {ai.chat(msg)}")
        
        # Сохранение
        print("\n5. Сохранение модели...")
        ai.save_model("models/synthesis_l1_trained.pth")
        
        print("\n🎉 ВСЁ ЗАВЕРШЕНО!")
    else:
        print("❌ Обучение не удалось")

if __name__ == "__main__":
    main()