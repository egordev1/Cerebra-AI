import os
import importlib
import logging
from typing import Dict, List, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Типы плагинов"""
    PROCESSOR = "processor"          # Обработка входных данных
    GENERATOR = "generator"          # Генерация ответов
    ENHANCER = "enhancer"            # Улучшение функционала
    INTEGRATION = "integration"      # Интеграция с внешними сервисами
    TOOL = "tool"                    # Вспомогательные инструменты

@dataclass
class PluginInfo:
    """Информация о плагине"""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str]
    enabled: bool = True

class BasePlugin(ABC):
    """Базовый класс для всех плагинов"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
    
    @abstractmethod
    def initialize(self):
        """Инициализация плагина"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Выполнение основной функции плагина"""
        pass
    
    def cleanup(self):
        """Очистка ресурсов плагина"""
        pass

class PluginManager:
    """Менеджер плагинов для Cerebra AI"""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.plugin_hooks: Dict[str, List[Callable]] = {
            'pre_process': [],
            'post_process': [],
            'response_generation': [],
            'error_handling': []
        }
        self.plugin_dirs = ['plugins', './cerebra/plugins']
    
    def register_plugin(self, plugin: BasePlugin, plugin_info: PluginInfo):
        """Регистрация плагина"""
        if plugin_info.name in self.plugins:
            logger.warning(f"Плагин {plugin_info.name} уже зарегистрирован, заменяем...")
        
        self.plugins[plugin_info.name] = plugin
        self.plugin_info[plugin_info.name] = plugin_info
        
        # Инициализируем плагин если он включен
        if plugin_info.enabled:
            try:
                plugin.initialize()
                logger.info(f"Плагин {plugin_info.name} зарегистрирован и инициализирован")
            except Exception as e:
                logger.error(f"Ошибка инициализации плагина {plugin_info.name}: {e}")
                plugin_info.enabled = False
    
    def load_plugin_from_file(self, file_path: str):
        """Загрузка плагина из файла"""
        try:
            # Получаем имя модуля из пути
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Динамически импортируем модуль
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Ищем классы плагинов в модуле
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    # Создаем экземпляр плагина
                    plugin_instance = attr(attr_name)
                    
                    # Создаем базовую информацию о плагине
                    plugin_info = PluginInfo(
                        name=attr_name,
                        version="1.0.0",
                        author="Unknown",
                        description=f"Plugin {attr_name}",
                        plugin_type=PluginType.TOOL,
                        dependencies=[]
                    )
                    
                    self.register_plugin(plugin_instance, plugin_info)
                    logger.info(f"Плагин {attr_name} загружен из {file_path}")
                    break
            
        except Exception as e:
            logger.error(f"Ошибка загрузки плагина из {file_path}: {e}")
    
    def load_plugins_from_directory(self, directory: str):
        """Загрузка всех плагинов из директории"""
        if not os.path.exists(directory):
            logger.warning(f"Директория плагинов {directory} не существует")
            return
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(root, file)
                    self.load_plugin_from_file(file_path)
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Выполнение плагина"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Плагин {plugin_name} не найден")
        
        plugin = self.plugins[plugin_name]
        if not self.plugin_info[plugin_name].enabled:
            raise ValueError(f"Плагин {plugin_name} отключен")
        
        return plugin.execute(*args, **kwargs)
    
    def enable_plugin(self, plugin_name: str):
        """Включение плагина"""
        if plugin_name in self.plugin_info:
            self.plugin_info[plugin_name].enabled = True
            logger.info(f"Плагин {plugin_name} включен")
    
    def disable_plugin(self, plugin_name: str):
        """Отключение плагина"""
        if plugin_name in self.plugin_info:
            self.plugin_info[plugin_name].enabled = False
            logger.info(f"Плагин {plugin_name} отключен")
    
    def get_plugin_info(self, plugin_name: str) -> PluginInfo:
        """Получение информации о плагине"""
        return self.plugin_info.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """Список всех зарегистрированных плагинов"""
        return list(self.plugins.keys())
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Регистрация хука"""
        if hook_name not in self.plugin_hooks:
            self.plugin_hooks[hook_name] = []
        self.plugin_hooks[hook_name].append(callback)
    
    def execute_hook(self, hook_name: str, *args, **kwargs):
        """Выполнение всех хуков для указанного события"""
        if hook_name in self.plugin_hooks:
            for callback in self.plugin_hooks[hook_name]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Ошибка выполнения хука {hook_name}: {e}")

# Примеры конкретных плагинов

class WebSearchPlugin(BasePlugin):
    """Плагин для веб-поиска"""
    
    def __init__(self):
        super().__init__("web_search", "1.0.0")
        self.search_providers = []
    
    def initialize(self):
        logger.info("Инициализация плагина веб-поиска")
        # Здесь можно инициализировать провайдеров поиска
    
    def execute(self, query: str, *args, **kwargs) -> str:
        from .web_search import search_web
        return search_web(query)

class CalculatorPlugin(BasePlugin):
    """Плагин для математических вычислений"""
    
    def __init__(self):
        super().__init__("calculator", "1.0.0")
    
    def initialize(self):
        logger.info("Инициализация плагина калькулятора")
    
    def execute(self, expression: str, *args, **kwargs) -> str:
        try:
            # Безопасное вычисление выражения
            allowed_chars = set('0123456789+-*/().% ')
            if not all(c in allowed_chars for c in expression):
                return "Ошибка: недопустимые символы в выражении"
            
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Результат: {result}"
        except Exception as e:
            return f"Ошибка вычисления: {str(e)}"

class DateTimePlugin(BasePlugin):
    """Плагин для работы с датой и временем"""
    
    def __init__(self):
        super().__init__("datetime", "1.0.0")
    
    def initialize(self):
        import datetime
        self.datetime_module = datetime
    
    def execute(self, action: str, *args, **kwargs) -> str:
        if action == "current_time":
            return f"Текущее время: {self.datetime_module.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elif action == "current_date":
            return f"Текущая дата: {self.datetime_module.date.today().strftime('%Y-%m-%d')}"
        else:
            return "Неизвестное действие для плагина даты/времени"

class FileOperationPlugin(BasePlugin):
    """Плагин для операций с файлами"""
    
    def __init__(self):
        super().__init__("file_operations", "1.0.0")
    
    def initialize(self):
        logger.info("Инициализация плагина файловых операций")
    
    def execute(self, operation: str, file_path: str, *args, **kwargs) -> str:
        try:
            if operation == "read":
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"Содержимое файла:\n{content}"
            elif operation == "write":
                content = kwargs.get('content', '')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"Файл {file_path} успешно записан"
            elif operation == "list":
                files = os.listdir(file_path)
                return f"Файлы в директории {file_path}:\n" + "\n".join(files)
            else:
                return "Неизвестная операция"
        except Exception as e:
            return f"Ошибка файловой операции: {str(e)}"

def create_default_plugin_manager() -> PluginManager:
    """Создание менеджера плагинов с базовыми плагинами"""
    pm = PluginManager()
    
    # Регистрируем базовые плагины
    plugins_with_info = [
        (WebSearchPlugin(), PluginInfo(
            name="web_search",
            version="1.0.0",
            author="Cerebra AI Team",
            description="Плагин для веб-поиска",
            plugin_type=PluginType.INTEGRATION,
            dependencies=[]
        )),
        (CalculatorPlugin(), PluginInfo(
            name="calculator",
            version="1.0.0",
            author="Cerebra AI Team",
            description="Плагин для математических вычислений",
            plugin_type=PluginType.TOOL,
            dependencies=[]
        )),
        (DateTimePlugin(), PluginInfo(
            name="datetime",
            version="1.0.0",
            author="Cerebra AI Team",
            description="Плагин для работы с датой и временем",
            plugin_type=PluginType.TOOL,
            dependencies=[]
        )),
        (FileOperationPlugin(), PluginInfo(
            name="file_operations",
            version="1.0.0",
            author="Cerebra AI Team",
            description="Плагин для операций с файлами",
            plugin_type=PluginType.TOOL,
            dependencies=[]
        ))
    ]
    
    for plugin, info in plugins_with_info:
        pm.register_plugin(plugin, info)
    
    # Загружаем дополнительные плагины из директории
    for plugin_dir in pm.plugin_dirs:
        pm.load_plugins_from_directory(plugin_dir)
    
    return pm