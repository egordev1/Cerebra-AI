# Экспорт главной модели
try:
    from .main_model import SynthesisL1
    __all__ = ['SynthesisL1']
except ImportError:
    __all__ = []