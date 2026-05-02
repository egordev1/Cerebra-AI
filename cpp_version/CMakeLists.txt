cmake_minimum_required(VERSION 3.18)
project(CerebraAI VERSION 2.0.0 LANGUAGES CXX)

# Стандарт C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Настройки компиляции
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Поиск LibTorch
find_package(Torch REQUIRED)

# Исходные файлы
set(SOURCES
    src/tokenizer.cpp
    src/gpt_model.cpp
    src/main_model.cpp
    src/cerebra_ai.cpp
)

# Заголовочные файлы
set(HEADERS
    include/tokenizer.hpp
    include/gpt_model.hpp
    include/main_model.hpp
    include/cerebra_ai.hpp
)

# Создание библиотеки
add_library(cerebra_lib STATIC ${SOURCES})
target_include_directories(cerebra_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cerebra_lib "${TORCH_LIBRARIES}")

# Основная программа
add_executable(cerebra_ai src/main.cpp)
target_link_libraries(cerebra_ai cerebra_lib "${TORCH_LIBRARIES}")

# Копирование зависимостей для Windows
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET cerebra_ai
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:cerebra_ai>)
endif()

# Установка
install(TARGETS cerebra_ai DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# Вывод информации
message(STATUS "Cerebra AI v${PROJECT_VERSION}")
message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX}")
message(STATUS "CUDA available: ${USE_CUDA}")
