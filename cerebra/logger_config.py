"""
Конфигурация логирования для Cerebra AI
Файл: logger_config.py - Настройка системы логирования, запись в файлы и консоль
"""
import logging
import os
from datetime import datetime

def setup_logger(name='cerebra', log_level=logging.INFO):
    """
    Настройка логгера для Cerebra AI
    
    Args:
        name: имя логгера
        log_level: уровень логирования
        
    Returns:
        logger: настроенный объект логгера
    """
    # Создаем директорию для логов, если её нет
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Имя файла лога с датой
    log_filename = os.path.join(log_dir, f"cerebra_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Настройка логгера
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Очищаем существующие обработчики
    logger.handlers = []
    
    # Обработчик для файла
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Глобальный логгер
logger = setup_logger()

