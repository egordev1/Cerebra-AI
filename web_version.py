#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cerebra AI - Веб-версия приложения
Файл: web_version.py - Точка входа для запуска веб-интерфейса
"""

import sys
import os
import logging
import threading
import time
from typing import Dict, Any

# Установка UTF-8 кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Добавляем текущую папку в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cerebra.web_interface import CerebraWebInterface
from cerebra.logger_config import logger

def main():
    """Главная функция для запуска веб-версии"""
    # Отдельная строка в консоли, как было запрошено
    print("🌐 Cerebra AI - ВЕБ-ВЕРСИЯ")
    print("🚀 Запуск веб-интерфейса Cerebra AI...")
    
    # Настройка параметров сервера
    host = os.getenv("CEREBRA_HOST", "0.0.0.0")
    port = int(os.getenv("CEREBRA_PORT", "8000"))
    
    print(f"🌐 Веб-интерфейс будет доступен по адресу: http://{host}:{port}")
    print("💡 Для остановки сервера используйте Ctrl+C")
    
    # Создаем и запускаем веб-интерфейс
    web_interface = CerebraWebInterface(host=host, port=port)
    
    try:
        # Запускаем сервер
        web_interface.run(debug=False)
    except KeyboardInterrupt:
        print("\n👋 Веб-сервер остановлен")
    except Exception as e:
        logger.error(f"Ошибка при запуске веб-сервера: {e}")
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()