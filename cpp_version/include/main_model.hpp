/**
 * @file main_model.hpp
 * @brief Главная модель Synthesis-L1
 * C++ реализация с использованием LibTorch
 */

#ifndef CEREBRA_MAIN_MODEL_HPP
#define CEREBRA_MAIN_MODEL_HPP

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace cerebra {

class SimpleTokenizer;
class GPTTransformer;

/**
 * @brief Главная модель Synthesis-L1 - современная GPT трансформерная модель
 */
class SynthesisL1 : public torch::nn::Module {
public:
    SynthesisL1(bool use_gpt = true);
    
    // Обработка текста и генерация ответа
    std::string process(const std::string& text);
    
    // Подготовка данных для обучения
    std::pair<std::vector<std::string>, std::vector<int>> prepare_data();
    
    // Обучение модели
    bool real_train(int64_t epochs = 10, int64_t batch_size = 4, double lr = 3e-4);
    
    // Получить информацию о модели
    std::unordered_map<std::string, int64_t> get_info() const;
    
    // Сохранение модели
    bool save(const std::string& path) const;
    
    // Загрузка модели
    bool load(const std::string& path, const torch::Device& device = torch::kCPU);

private:
    std::string model_id_;
    std::string version_;
    bool use_gpt_;
    bool is_trained_;
    
    std::unique_ptr<SimpleTokenizer> tokenizer_;
    std::unique_ptr<GPTTransformer> gpt_model_;
    
    // Fallback словарь для старой LSTM модели
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> reverse_vocab_;
    
    torch::nn::Embedding embedding_{nullptr};
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear fc_{nullptr};
    
    // Базовые ответы для fallback
    std::unordered_map<std::string, std::string> fallback_responses_;
    
    void init_fallback_model();
    std::string fallback_process(const std::string& text);
};

} // namespace cerebra

#endif // CEREBRA_MAIN_MODEL_HPP
