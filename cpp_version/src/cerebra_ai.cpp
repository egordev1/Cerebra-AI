/**
 * @file cerebra_ai.cpp
 * @brief Реализация ядра Cerebra AI
 */

#include "cerebra_ai.hpp"
#include "main_model.hpp"
#include <iostream>
#include <algorithm>

namespace cerebra {

CerebraAI::CerebraAI() 
    : name_("CerebraAI"), version_("2.0.0"), device_(torch::kCPU) {
    
    // Определение устройства
    if (torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA);
        std::cout << "💾 GPU доступен: CUDA" << std::endl;
    } else if (torch::mps::is_available()) {
        device_ = torch::Device(torch::kMPS);
        std::cout << "💾 GPU доступен: MPS (Apple Silicon)" << std::endl;
    }
    
    std::cout << "🧠 Запущена " << name_ << " AI System v" << version_ << std::endl;
    std::cout << "📊 Устройство: " << device_ << std::endl;
}

bool CerebraAI::load_model(const std::string& model_name, bool quantize, 
                            const std::string& quantization_method) {
    try {
        std::cout << "\n📦 Загрузка модели " << model_name << "..." << std::endl;
        
        if (model_name == "Synthesis-L1") {
            active_model_ = std::make_unique<SynthesisL1>(true);
            active_model_->to(device_);
            
            std::cout << "✅ Загружена модель: " << model_name << std::endl;
            
            // Применяем квантование если нужно
            if (quantize) {
                std::cout << "ℹ️  Квантование пока не реализовано в C++ версии" << std::endl;
            }
            
            return true;
            
        } else if (model_name == "Synthesis-L2" || model_name == "Synthesis-L3") {
            std::cout << "⚠️  Модели Synthesis-L2/L3 пока не реализованы в C++ версии" << std::endl;
            return false;
            
        } else {
            std::cerr << "❌ Модель " << model_name << " не найдена" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка при загрузке модели: " << e.what() << std::endl;
        return false;
    }
}

std::string CerebraAI::chat(const std::string& message, bool use_web_search, bool use_plugins) {
    if (!active_model_) {
        return "⚠️  Сначала загрузите модель!";
    }
    
    // Проверяем, нужно ли использовать плагины
    if (use_plugins) {
        auto response = process_plugin_request(message);
        if (!response.empty()) {
            return response;
        }
    }
    
    // Веб-поиск если нужно (пока не реализован в C++ версии)
    if (use_web_search) {
        std::cout << "⚠️  Веб-поиск пока не реализован в C++ версии" << std::endl;
    }
    
    // Генерация ответа моделью
    return active_model_->process(message);
}

std::string CerebraAI::generate_response(const std::string& message,
                                          const std::string& model_type,
                                          bool use_web_search,
                                          bool use_plugins) {
    // Проверяем, нужно ли сменить модель
    if (!active_model_) {
        load_model("Synthesis-" + model_type);
    }
    
    return chat(message, use_web_search, use_plugins);
}

std::string CerebraAI::process_plugin_request(const std::string& message) {
    std::string message_lower = message;
    std::transform(message_lower.begin(), message_lower.end(), message_lower.begin(), ::tolower);
    
    // Проверяем команды плагинов
    if (message.find('!') == 0) {
        size_t space_pos = message.find(' ');
        std::string command = (space_pos != std::string::npos) ? 
                              message.substr(1, space_pos - 1) : 
                              message.substr(1);
        
        if (command == "calc" || message_lower.find("калькулятор") != std::string::npos ||
            message_lower.find("посчитай") != std::string::npos) {
            return "⚠️  Калькулятор пока не реализован в C++ версии";
        }
        
        if (command == "time" || message_lower.find("время") != std::string::npos) {
            return "⚠️  Время пока не реализовано в C++ версии";
        }
        
        if (command == "date" || message_lower.find("дата") != std::string::npos) {
            return "⚠️  Дата пока не реализована в C++ версии";
        }
    }
    
    return "";
}

bool CerebraAI::quantize_model(const std::string& method) {
    if (!active_model_) {
        std::cerr << "Нет активной модели для квантования" << std::endl;
        return false;
    }
    
    std::cout << "⚠️  Квантование пока не реализовано в C++ версии" << std::endl;
    return false;
}

std::string CerebraAI::info() const {
    std::string info_text = "\n🧠 " + name_ + " AI System v" + version_ + "\n";
    info_text += "📊 Устройство: " + std::string(device_.str()) + "\n";
    info_text += "⚡ GPU Ускорение: " + std::string(torch::cuda::is_available() ? "Доступно" : "Недоступно") + "\n\n";
    
    info_text += "Доступные модели:\n";
    info_text += "• Synthesis-L1 - GPT трансформерная модель (текстовая генерация)\n";
    info_text += "• Synthesis-L2 - Продвинутая GPT-3 подобная модель (не реализована)\n";
    info_text += "• Synthesis-L3 - Ещё более продвинутая модель (не реализована)\n\n";
    
    info_text += "Доступные команды плагинов:\n";
    info_text += "  !calc <выражение> или 'посчитай' - Математические вычисления\n";
    info_text += "  !time или 'время' - Текущее время\n";
    info_text += "  !date или 'дата' - Текущая дата\n";
    info_text += "  !search <запрос> или 'найди' - Веб-поиск\n";
    
    if (active_model_) {
        auto model_info = active_model_->get_info();
        info_text += "\n🎯 Активная модель: Synthesis-L1";
        info_text += "\n📈 Параметров: " + std::to_string(model_info.at("parameters"));
    }
    
    return info_text;
}

bool CerebraAI::save_model(const std::string& path) const {
    if (!active_model_) {
        std::cerr << "Нет активной модели для сохранения" << std::endl;
        return false;
    }
    
    try {
        return active_model_->save(path);
    } catch (const std::exception& e) {
        std::cerr << "Ошибка при сохранении модели: " << e.what() << std::endl;
        return false;
    }
}

bool CerebraAI::load_model_from_file(const std::string& path) {
    if (!active_model_) {
        std::cerr << "Сначала загрузите модель" << std::endl;
        return false;
    }
    
    try {
        return active_model_->load(path, device_);
    } catch (const std::exception& e) {
        std::cerr << "Ошибка при загрузке модели: " << e.what() << std::endl;
        return false;
    }
}

} // namespace cerebra
