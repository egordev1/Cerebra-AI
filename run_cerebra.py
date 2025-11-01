#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой запуск Cerebra AI
"""

import sys
import io

# Установка UTF-8 кодировки для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from cerebra import ai
from cerebra.logger_config import logger

def main():
    logger.info("🚀 Запуск Cerebra AI...")
    
    # Загрузка модели
    model = ai.load_model("Synthesis-L1")
    
    # Информация
    logger.info(ai.info())
    
    # Чат
    logger.info("\n💬 Тестируем чат:")
    try:
        while True:
            user_input = input("\n👤 Вы: ").strip()
            if not user_input:  # Пустой ввод
                continue
            if user_input.lower() in ['выход', 'exit', 'quit']:
                break
            logger.debug(f"Пользовательский ввод: {user_input}")
            response = ai.chat(user_input)
            print(f"🤖 Cerebra: {response}")
            logger.debug(f"Ответ модели: {response}")
    except (KeyboardInterrupt, EOFError):
        logger.info("\n\n👋 До свидания!")
        print("\n\n👋 До свидания!")

if __name__ == "__main__":
    main()