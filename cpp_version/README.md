# Cerebra AI - C++ Version

Реконструкция проекта Cerebra AI на языке программирования C++ с использованием библиотеки LibTorch.

## 📋 Описание

Cerebra AI - это система искусственного интеллекта с трансформерными моделями для обработки естественного языка. Данная версия представляет собой полную реконструкцию оригинального Python-проекта на C++.

## 🔧 Требования

### Обязательные
- **Компилятор**: GCC 9+ или Clang 10+ или MSVC 2019+
- **CMake**: 3.18 или выше
- **LibTorch**: 1.9 или выше (PyTorch C++ API)
- **Стандарт C++**: C++17

### Опциональные
- **CUDA**: 11.0+ для GPU ускорения
- **cuDNN**: Для ускорения свёрточных операций

## 📦 Установка LibTorch

### Linux (CPU)
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu-linux-x86_64.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu-linux-x86_64.zip
export Torch_DIR=$(pwd)/libtorch/share/cmake/Torch
```

### Linux (CUDA)
```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118-linux-x86_64.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118-linux-x86_64.zip
export Torch_DIR=$(pwd)/libtorch/share/cmake/Torch
```

### Windows
Скачайте LibTorch с официального сайта: https://pytorch.org/get-started/locally/

### macOS
```bash
brew install libtorch
```

## 🏗️ Сборка проекта

```bash
cd cpp_version
mkdir build && cd build

# Настройка (укажите путь к LibTorch если нужно)
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch

# Сборка
cmake --build . --config Release

# Запуск
./cerebra_ai
```

## 📁 Структура проекта

```
cpp_version/
├── include/                    # Заголовочные файлы
│   ├── tokenizer.hpp          # Токенизатор
│   ├── gpt_model.hpp          # GPT трансформер модель
│   ├── main_model.hpp         # Главная модель Synthesis-L1
│   └── cerebra_ai.hpp         # Ядро системы
├── src/                        # Исходные файлы
│   ├── tokenizer.cpp
│   ├── gpt_model.cpp
│   ├── main_model.cpp
│   ├── cerebra_ai.cpp
│   └── main.cpp               # Главная программа
├── build/                      # Директория сборки
├── CMakeLists.txt             # Конфигурация CMake
└── README.md                  # Этот файл
```

## 🎯 Основные компоненты

### 1. Токенизатор (`tokenizer.hpp/cpp`)
- BPE-подобная токенизация
- Поддержка специальных токенов (<PAD>, <UNK>, <BOS>, <EOS>)
- Сохранение и загрузка словаря

### 2. GPT Модель (`gpt_model.hpp/cpp`)
- Позиционное кодирование
- Multi-Head Self-Attention
- Transformer блоки
- Генерация текста с top-k и top-p sampling

### 3. Главная модель (`main_model.hpp/cpp`)
- Synthesis-L1 архитектура
- Поддержка GPT и LSTM режимов
- Обучение и инференс

### 4. Ядро системы (`cerebra_ai.hpp/cpp`)
- Управление моделями
- Диалоговый интерфейс
- Система плагинов (заглушка)

## 💡 Пример использования

```cpp
#include "cerebra_ai.hpp"

using namespace cerebra;

int main() {
    // Создание AI системы
    CerebraAI ai;
    
    // Загрузка модели
    ai.load_model("Synthesis-L1");
    
    // Чат
    std::string response = ai.chat("Привет! Как дела?");
    std::cout << response << std::endl;
    
    return 0;
}
```

## 🔍 Отличия от Python версии

### Реализовано
- ✅ Токенизатор
- ✅ GPT трансформер модель
- ✅ Synthesis-L1 модель
- ✅ Базовый чат-интерфейс
- ✅ Сохранение/загрузка моделей

### В разработке
- ⏳ Полноценное обучение модели
- ⏳ Система плагинов
- ⏳ Веб-поиск
- ⏳ Квантование моделей
- ⏳ Модели Synthesis-L2/L3
- ⏳ Диалоговое обучение

## 🚀 Производительность

C++ версия обеспечивает:
- Более быстрый инференс благодаря компиляции
- Меньшее потребление памяти
- Возможность оптимизации под конкретное железо
- Интеграция с другими C++ проектами

## 📝 Лицензия

Лицензия аналогична оригинальному проекту Cerebra AI.

## 👥 Авторы

Реконструкция на C++ выполнена на основе оригинального Python проекта Cerebra AI.

## 🤝 Вклад в проект

Проект открыт для улучшений. Feel free to submit issues and enhancement requests!

## 📞 Контакты

Для вопросов и предложений обращайтесь через GitHub Issues.
