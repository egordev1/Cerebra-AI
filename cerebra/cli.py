import argparse
import sys
import os
from typing import Optional
import logging

# Добавляем корневую директорию в путь Python для правильного импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .core import CerebraAI
from .web_interface import create_web_interface

logger = logging.getLogger(__name__)

class CerebraCLI:
    """Командный интерфейс для Cerebra AI"""
    
    def __init__(self):
        self.ai = CerebraAI()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Создание парсера аргументов командной строки"""
        parser = argparse.ArgumentParser(
            description="Cerebra AI - Мощная система искусственного интеллекта",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Примеры использования:
  python -m cerebra.cli chat                    # Интерактивный чат
  python -m cerebra.cli chat -m "Привет!"      # Одиночное сообщение
  python -m cerebra.cli chat -t L2             # Использовать модель L2
  python -m cerebra.cli info                    # Информация о системе
  python -m cerebra.cli web                     # Запустить веб-интерфейс
  python -m cerebra.cli train --epochs 20       # Обучить модель
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
        
        # Команда чата
        chat_parser = subparsers.add_parser('chat', help='Интерактивный чат с AI')
        chat_parser.add_argument('-m', '--message', type=str, help='Сообщение для AI')
        chat_parser.add_argument('-t', '--model', type=str, default='L1', 
                                choices=['L1', 'L2', 'L3'], help='Выбор модели (L1, L2, L3)')
        chat_parser.add_argument('--no-web-search', action='store_true', 
                                help='Отключить веб-поиск')
        chat_parser.add_argument('--no-plugins', action='store_true', 
                                help='Отключить плагины')
        chat_parser.add_argument('--quantize', action='store_true', 
                                help='Использовать квантование модели')
        
        # Команда информации
        info_parser = subparsers.add_parser('info', help='Информация о системе')
        
        # Команда обучения
        train_parser = subparsers.add_parser('train', help='Обучение модели')
        train_parser.add_argument('--epochs', type=int, default=10, help='Количество эпох')
        train_parser.add_argument('--batch-size', type=int, default=4, help='Размер батча')
        
        # Команда веб-интерфейса
        web_parser = subparsers.add_parser('web', help='Запуск веб-интерфейса')
        web_parser.add_argument('--host', type=str, default='0.0.0.0', help='Хост для веб-сервера')
        web_parser.add_argument('--port', type=int, default=8000, help='Порт для веб-сервера')
        
        # Команда квантования
        quantize_parser = subparsers.add_parser('quantize', help='Квантование модели')
        quantize_parser.add_argument('--method', type=str, default='dynamic',
                                   choices=['dynamic', 'static', 'qat', 'mixed'],
                                   help='Метод квантования')
        
        return parser
    
    def run_chat(self, args):
        """Запуск чат-режима"""
        # Загружаем модель
        self.ai.load_model(f"Synthesis-{args.model}", 
                          quantize=args.quantize, 
                          quantization_method='dynamic' if args.quantize else 'dynamic')
        
        if args.message:
            # Одиночное сообщение
            response = self.ai.chat(
                args.message,
                use_web_search=not args.no_web_search,
                use_plugins=not args.no_plugins
            )
            print(f"🤖 AI: {response}")
        else:
            # Интерактивный режим
            print(f"🧠 Загружена модель: {args.model}")
            print("💬 Начало интерактивного чата (введите 'exit' для выхода)")
            print("💡 Подсказки:")
            print("   !calc 2+2 - математические вычисления")
            print("   !time - текущее время")
            print("   !date - текущая дата")
            print("   !search запрос - веб-поиск")
            print()
            
            while True:
                try:
                    user_input = input("👤 Вы: ").strip()
                    if user_input.lower() in ['exit', 'quit', 'выйти', 'q']:
                        print("👋 Пока!")
                        break
                    if user_input:
                        response = self.ai.chat(
                            user_input,
                            use_web_search=not args.no_web_search,
                            use_plugins=not args.no_plugins
                        )
                        print(f"🤖 AI: {response}")
                        print()
                except KeyboardInterrupt:
                    print("\n👋 Пока!")
                    break
                except EOFError:
                    print("\n👋 Пока!")
                    break
    
    def run_info(self, args):
        """Вывод информации о системе"""
        info = self.ai.info()
        print(info)
    
    def run_train(self, args):
        """Запуск обучения модели"""
        print(f"🚀 Начало обучения модели на {args.epochs} эпох...")
        success = self.ai.real_training(epochs=args.epochs, batch_size=args.batch_size)
        if success:
            print("🎉 Обучение завершено успешно!")
        else:
            print("❌ Ошибка во время обучения")
    
    def run_web(self, args):
        """Запуск веб-интерфейса"""
        print(f"🌐 Запуск веб-интерфейса на {args.host}:{args.port}")
        web_interface = create_web_interface(host=args.host, port=args.port)
        web_interface.run()
    
    def run_quantize(self, args):
        """Квантование модели"""
        print(f"⚡ Квантование модели методом {args.method}...")
        success = self.ai.quantize_model(method=args.method)
        if success:
            print("✅ Квантование завершено успешно!")
        else:
            print("❌ Ошибка квантования")
    
    def run(self, args=None):
        """Запуск CLI"""
        if args is None:
            args = self.parser.parse_args()
        
        if args.command == 'chat':
            self.run_chat(args)
        elif args.command == 'info':
            self.run_info(args)
        elif args.command == 'train':
            self.run_train(args)
        elif args.command == 'web':
            self.run_web(args)
        elif args.command == 'quantize':
            self.run_quantize(args)
        else:
            self.parser.print_help()

def main():
    """Основная функция CLI"""
    cli = CerebraCLI()
    cli.run()

if __name__ == "__main__":
    main()