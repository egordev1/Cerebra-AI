/**
 * @file cerebra_ai.hpp
 * @brief Ядро системы Cerebra AI
 * Главный контроллер AI системы, управляет моделями, обучением, чатом
 */

#ifndef CEREBRA_AI_HPP
#define CEREBRA_AI_HPP

#include <torch/torch.h>
#include <string>
#include <memory>
#include <unordered_map>

namespace cerebra {

class SynthesisL1;

/**
 * @brief Главное ядро Cerebra AI системы
 */
class CerebraAI {
public:
    CerebraAI();
    
    // Загрузка модели
    bool load_model(const std::string& model_name = "Synthesis-L1", 
                    bool quantize = false, 
                    const std::string& quantization_method = "dynamic");
    
    // Общение с ИИ
    std::string chat(const std::string& message, 
                     bool use_web_search = false, 
                     bool use_plugins = true);
    
    // Генерация ответа с возможностью выбора модели
    std::string generate_response(const std::string& message,
                                  const std::string& model_type = "L1",
                                  bool use_web_search = false,
                                  bool use_plugins = true);
    
    // Квантование модели
    bool quantize_model(const std::string& method = "dynamic");
    
    // Получить информацию о системе
    std::string info() const;
    
    // Сохранение модели
    bool save_model(const std::string& path) const;
    
    // Загрузка модели из файла
    bool load_model_from_file(const std::string& path);

private:
    std::string name_;
    std::string version_;
    torch::Device device_;
    
    std::unique_ptr<SynthesisL1> active_model_;
    
    // Обработка запросов к плагинам
    std::string process_plugin_request(const std::string& message);
};

} // namespace cerebra

#endif // CEREBRA_AI_HPP
