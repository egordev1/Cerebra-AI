/**
 * @file main.cpp
 * @brief Главная программа Cerebra AI на C++
 * Демонстрация работы C++ версии
 */

#include <iostream>
#include <string>
#include <sstream>
#include "cerebra_ai.hpp"

using namespace cerebra;

void print_menu() {
    std::cout << "\n========================================\n";
    std::cout << "       Cerebra AI - Главное Меню\n";
    std::cout << "========================================\n";
    std::cout << "1. Загрузить модель Synthesis-L1\n";
    std::cout << "2. Обучить модель\n";
    std::cout << "3. Чат с ИИ\n";
    std::cout << "4. Информация о системе\n";
    std::cout << "5. Сохранить модель\n";
    std::cout << "6. Загрузить модель из файла\n";
    std::cout << "0. Выход\n";
    std::cout << "========================================\n";
    std::cout << "> ";
}

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║     Cerebra AI v2.0.0 (C++ Version)   ║\n";
    std::cout << "║           Реконструкция на C++         ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    CerebraAI ai;
    bool model_loaded = false;
    
    while (true) {
        print_menu();
        
        int choice;
        if (!(std::cin >> choice)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Неверный ввод! Попробуйте снова.\n";
            continue;
        }
        std::cin.ignore(); // Ignore newline
        
        switch (choice) {
            case 1: {
                std::cout << "\nЗагрузка модели Synthesis-L1...\n";
                model_loaded = ai.load_model("Synthesis-L1");
                if (model_loaded) {
                    std::cout << "✅ Модель успешно загружена!\n";
                } else {
                    std::cout << "❌ Ошибка загрузки модели!\n";
                }
                break;
            }
            
            case 2: {
                if (!model_loaded) {
                    std::cout << "⚠️  Сначала загрузите модель!\n";
                    break;
                }
                std::cout << "\nНачинаю обучение модели...\n";
                // Здесь должна быть реализация обучения
                std::cout << "ℹ️  Обучение пока не полностью реализовано в демо-версии\n";
                break;
            }
            
            case 3: {
                if (!model_loaded) {
                    std::cout << "⚠️  Сначала загрузите модель!\n";
                    break;
                }
                
                std::cout << "\n╔════════════════════════════════════════╗\n";
                std::cout << "║         Чат с Cerebra AI              ║\n";
                std::cout << "║    (введите 'выход' для возврата)     ║\n";
                std::cout << "╚════════════════════════════════════════╝\n\n";
                
                while (true) {
                    std::cout << "Вы: ";
                    std::string message;
                    std::getline(std::cin, message);
                    
                    if (message == "выход" || message == "exit" || message == "quit") {
                        break;
                    }
                    
                    if (message.empty()) continue;
                    
                    std::string response = ai.chat(message);
                    std::cout << "\nCerebra: " << response << "\n\n";
                }
                break;
            }
            
            case 4: {
                std::cout << ai.info();
                break;
            }
            
            case 5: {
                if (!model_loaded) {
                    std::cout << "⚠️  Сначала загрузите модель!\n";
                    break;
                }
                
                std::cout << "\nВведите путь для сохранения модели: ";
                std::string path;
                std::getline(std::cin, path);
                
                if (ai.save_model(path)) {
                    std::cout << "✅ Модель сохранена: " << path << "\n";
                } else {
                    std::cout << "❌ Ошибка сохранения модели!\n";
                }
                break;
            }
            
            case 6: {
                std::cout << "\nВведите путь к файлу модели: ";
                std::string path;
                std::getline(std::cin, path);
                
                if (ai.load_model_from_file(path)) {
                    model_loaded = true;
                    std::cout << "✅ Модель загружена из: " << path << "\n";
                } else {
                    std::cout << "❌ Ошибка загрузки модели!\n";
                }
                break;
            }
            
            case 0: {
                std::cout << "\n👋 До свидания! Удачи!\n";
                return 0;
            }
            
            default: {
                std::cout << "❌ Неверный выбор! Попробуйте снова.\n";
                break;
            }
        }
    }
    
    return 0;
}
