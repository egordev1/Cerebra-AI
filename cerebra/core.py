import torch
import os

class Cerebra:
    def __init__(self):
        self.name = "Cerebra"
        self.version = "1.0.0"
        self.active_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🧠 Запущена {self.name} AI System")
        print(f"📊 Устройство: {self.device}")
    
    def load_model(self, model_name="Synthesis-L1"):
        if model_name == "Synthesis-L1":
            from .models.language import SynthesisL1
            self.active_model = SynthesisL1().to(self.device)
            print(f"✅ Загружена модель: {model_name}")
        else:
            print(f"❌ Модель {model_name} не найдена")
            return None
        return self.active_model
    
    def chat(self, message):
        if not self.active_model:
            return "⚠️ Сначала загрузите модель: ai.load_model()"
        return self.active_model.process(message)
    
    def real_training(self, epochs=5):
        if not self.active_model:
            print("❌ Нет активной модели")
            return False
        
        try:
            if hasattr(self.active_model, 'real_train'):
                print(f"🚀 Обучение на {epochs} эпох...")
                success = self.active_model.real_train(epochs=epochs)
                
                if success:
                    print("🎉 Обучение успешно завершено!")
                    # Тестируем после обучения
                    test_texts = ["как дела", "все работает", "что нового", "хорошая работа"]
                    print("\n🧪 Тест после обучения:")
                    for text in test_texts:
                        response = self.chat(text)
                        print(f"   '{text}' -> {response}")
                return success
            else:
                print("❌ Модель не поддерживает обучение")
                return False
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False
    
    def save_model(self, path="models/synthesis_l1.pth"):
        if not self.active_model:
            print("❌ Нет модели для сохранения")
            return False
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.active_model.state_dict(), path)
        print(f"💾 Модель сохранена: {path}")
        return True
    
    def info(self):
        info_text = f"""
🧠 {self.name} AI System v{self.version}
📊 Устройство: {self.device}

Доступные модели:
• Synthesis-L1 - текстовая модель
• Synthesis-L2 - в разработке
• Synthesis-L3 - в разработке
"""
        if self.active_model:
            model_info = self.active_model.get_info()
            info_text += f"\n🎯 Активная модель: {model_info['model_id']}"
            info_text += f"\n📈 Параметров: {model_info['parameters']:,}"
        
        return info_text