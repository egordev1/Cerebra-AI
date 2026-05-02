/**
 * @file gpt_model.hpp
 * @brief GPT-like трансформер модель для генерации текста
 * C++ реализация с использованием LibTorch
 */

#ifndef CEREBRA_GPT_MODEL_HPP
#define CEREBRA_GPT_MODEL_HPP

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

namespace cerebra {

// Forward declaration
class SimpleTokenizer;

/**
 * @brief Позиционное кодирование для трансформера
 */
class PositionalEncoding : public torch::nn::Module {
public:
    PositionalEncoding(int64_t d_model, int64_t max_len = 5000);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::Tensor pe_;
};

/**
 * @brief Multi-Head Self-Attention механизм
 */
class MultiHeadAttention : public torch::nn::Module {
public:
    MultiHeadAttention(int64_t d_model, int64_t n_heads);
    torch::Tensor forward(torch::Tensor query, torch::Tensor key, 
                          torch::Tensor value, torch::Tensor mask = nullptr);

private:
    int64_t d_model_;
    int64_t n_heads_;
    int64_t d_k_;
    
    torch::nn::Linear W_q_{nullptr};
    torch::nn::Linear W_k_{nullptr};
    torch::nn::Linear W_v_{nullptr};
    torch::nn::Linear W_o_{nullptr};
    
    torch::Tensor scaled_dot_product_attention(
        torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask);
};

/**
 * @brief Один блок трансформера
 */
class TransformerBlock : public torch::nn::Module {
public:
    TransformerBlock(int64_t d_model, int64_t n_heads, int64_t d_ff, double dropout = 0.1);
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = nullptr);

private:
    MultiHeadAttention attention_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::Sequential feed_forward{nullptr};
};

/**
 * @brief GPT-like трансформер модель для генерации текста
 */
class GPTTransformer : public torch::nn::Module {
public:
    GPTTransformer(int64_t vocab_size = 10000, int64_t d_model = 512,
                   int64_t n_heads = 8, int64_t n_layers = 6,
                   int64_t d_ff = 2048, int64_t max_seq_len = 512,
                   double dropout = 0.1);
    
    // Forward pass
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = nullptr);
    
    // Генерация текста на основе промпта
    std::string generate(SimpleTokenizer& tokenizer, const std::string& prompt,
                        int64_t max_length = 100, double temperature = 0.8,
                        int64_t top_k = 50, double top_p = 0.9);
    
    // Создание каузальной маски для GPT
    torch::Tensor create_causal_mask(int64_t seq_len, const torch::Device& device);
    
    // Получить информацию о модели
    std::unordered_map<std::string, int64_t> get_info() const;
    
    // Сохранение модели
    bool save(const std::string& path) const;
    
    // Загрузка модели
    bool load(const std::string& path, const torch::Device& device = torch::kCPU);

private:
    std::string model_id_;
    std::string version_;
    int64_t vocab_size_;
    int64_t d_model_;
    int64_t max_seq_len_;
    
    torch::nn::Embedding token_embedding_{nullptr};
    PositionalEncoding positional_encoding_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
    torch::nn::ModuleList transformer_blocks{nullptr};
    torch::nn::LayerNorm layer_norm_{nullptr};
    torch::nn::Linear lm_head_{nullptr};
    
    void init_weights();
};

} // namespace cerebra

#endif // CEREBRA_GPT_MODEL_HPP
