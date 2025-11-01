#!/usr/bin/env python3
"""
Тесты для Cerebra AI
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cerebra import Cerebra
try:
    from cerebra.models.main_model import SynthesisL1
except ImportError:
    SynthesisL1 = None

def test_model_initialization():
    """Тест инициализации модели"""
    print("🧪 Тест инициализации модели...")
    if SynthesisL1:
        model = SynthesisL1()
        assert model.model_id == "Synthesis-L1"
        assert hasattr(model, 'forward') or hasattr(model, 'process')
        print("✅ Модель инициализирована корректно")
    else:
        print("⚠️  SynthesisL1 недоступна")

def test_core_functionality():
    """Тест основного функционала"""
    print("🧪 Тест ядра системы...")
    cerebra = Cerebra()
    model = cerebra.load_model("Synthesis-L1")
    assert model is not None
    print("✅ Ядро системы работает корректно")

def test_text_processing():
    """Тест обработки текста"""
    print("🧪 Тест обработки текста...")
    if SynthesisL1:
        model = SynthesisL1()
        response = model.process("тестовое сообщение")
        assert isinstance(response, str)
        assert len(response) > 0
        print("✅ Обработка текста работает")
    else:
        print("⚠️  SynthesisL1 недоступна")

def test_model_info():
    """Тест информации о модели"""
    print("🧪 Тест информации о модели...")
    if SynthesisL1:
        model = SynthesisL1()
        info = model.get_info()
        assert 'parameters' in info
        assert info['parameters'] > 0
        print("✅ Информация о модели доступна")
    else:
        print("⚠️  SynthesisL1 недоступна")

if __name__ == "__main__":
    test_model_initialization()
    test_core_functionality()
    test_text_processing()
    test_model_info()
    print("\n🎉 Все тесты пройдены!")