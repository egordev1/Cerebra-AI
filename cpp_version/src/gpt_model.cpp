/**
 * @file gpt_model.cpp
 * @brief Реализация GPT трансформер модели
 */

#include "gpt_model.hpp"
#include "tokenizer.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace cerebra {

// ============================================================================
// PositionalEncoding
// ============================================================================

PositionalEncoding::PositionalEncoding(int64_t d_model, int64_t max_len) {
    auto pe = torch::zeros({max_len, d_model});
    auto position = torch::arange(0, max_len).unsqueeze(1).to(torch::kFloat);
    auto div_term = torch::exp(
        torch::arange(0, d_model, 2).to(torch::kFloat) * 
        (-std::log(10000.0) / d_model)
    );
    
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(None, None, 2)}, 
                  torch::sin(position * div_term));
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, None, 2)}, 
                  torch::cos(position * div_term));
    
    pe_ = pe.unsqueeze(1).detach();
    register_buffer("pe", pe_);
}

torch::Tensor PositionalEncoding::forward(torch::Tensor x) {
    return x + pe_.index({torch::indexing::Slice(None, x.size(0)), torch::indexing::Slice()});
}

// ============================================================================
// MultiHeadAttention
// ============================================================================

MultiHeadAttention::MultiHeadAttention(int64_t d_model, int64_t n_heads) 
    : d_model_(d_model), n_heads_(n_heads), d_k_(d_model / n_heads) {
    
    W_q_ = register_module("W_q", torch::nn::Linear(d_model, d_model));
    W_k_ = register_module("W_k", torch::nn::Linear(d_model, d_model));
    W_v_ = register_module("W_v", torch::nn::Linear(d_model, d_model));
    W_o_ = register_module("W_o", torch::nn::Linear(d_model, d_model));
}

torch::Tensor MultiHeadAttention::scaled_dot_product_attention(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask) {
    
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(d_k_);
    
    if (mask.defined()) {
        scores = scores.masked_fill(mask == 0, -1e9);
    }
    
    auto attention_weights = torch::softmax(scores, -1);
    return torch::matmul(attention_weights, V);
}

torch::Tensor MultiHeadAttention::forward(torch::Tensor query, torch::Tensor key, 
                                           torch::Tensor value, torch::Tensor mask) {
    
    int64_t batch_size = query.size(0);
    
    auto Q = W_q_->forward(query)
        .view({batch_size, -1, n_heads_, d_k_})
        .transpose(1, 2);
    auto K = W_k_->forward(key)
        .view({batch_size, -1, n_heads_, d_k_})
        .transpose(1, 2);
    auto V = W_v_->forward(value)
        .view({batch_size, -1, n_heads_, d_k_})
        .transpose(1, 2);
    
    auto attention_output = scaled_dot_product_attention(Q, K, V, mask);
    
    attention_output = attention_output.transpose(1, 2).contiguous()
        .view({batch_size, -1, d_model_});
    
    return W_o_->forward(attention_output);
}

// ============================================================================
// TransformerBlock
// ============================================================================

TransformerBlock::TransformerBlock(int64_t d_model, int64_t n_heads, int64_t d_ff, double dropout) {
    attention_ = register_module("attention", MultiHeadAttention(d_model, n_heads));
    norm1_ = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions(d_model)));
    norm2_ = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions(d_model)));
    
    feed_forward = torch::nn::Sequential(
        torch::nn::Linear(d_model, d_ff),
        torch::nn::GELU(),
        torch::nn::Dropout(dropout),
        torch::nn::Linear(d_ff, d_model),
        torch::nn::Dropout(dropout)
    );
    register_module("feed_forward", feed_forward);
}

torch::Tensor TransformerBlock::forward(torch::Tensor x, torch::Tensor mask) {
    // Self-attention с residual connection
    auto attn_output = attention_->forward(x, x, x, mask);
    x = norm1_->forward(x + attn_output);
    
    // Feed-forward с residual connection
    auto ff_output = feed_forward->forward(x);
    x = norm2_->forward(x + ff_output);
    
    return x;
}

// ============================================================================
// GPTTransformer
// ============================================================================

GPTTransformer::GPTTransformer(int64_t vocab_size, int64_t d_model, int64_t n_heads,
                                int64_t n_layers, int64_t d_ff, int64_t max_seq_len,
                                double dropout)
    : model_id_("Synthesis-L1-GPT"), version_("2.0.0"),
      vocab_size_(vocab_size), d_model_(d_model), max_seq_len_(max_seq_len) {
    
    token_embedding_ = register_module("token_embedding", 
                                        torch::nn::Embedding(vocab_size, d_model));
    positional_encoding_ = register_module("positional_encoding", 
                                            PositionalEncoding(d_model, max_seq_len));
    dropout_ = register_module("dropout", torch::nn::Dropout(dropout));
    
    transformer_blocks = torch::nn::ModuleList();
    for (int64_t i = 0; i < n_layers; ++i) {
        transformer_blocks->push_back(TransformerBlock(d_model, n_heads, d_ff, dropout));
    }
    register_module("transformer_blocks", transformer_blocks);
    
    layer_norm_ = register_module("layer_norm", 
                                   torch::nn::LayerNorm(torch::nn::LayerNormOptions(d_model)));
    lm_head_ = register_module("lm_head", torch::nn::Linear(d_model, vocab_size));
    
    init_weights();
    
    std::cout << "🎯 Создана " << model_id_ << " v" << version_ << std::endl;
    std::cout << "   Параметров: " << parameters(true).size() << std::endl;
}

void GPTTransformer::init_weights() {
    for (auto& module : modules()) {
        if (auto linear = dynamic_cast<torch::nn::Linear*>(module.get())) {
            torch::nn::init::normal_(linear->weight, 0.0, 0.02);
            if (linear->bias.defined()) {
                torch::nn::init::zeros_(linear->bias);
            }
        } else if (auto embedding = dynamic_cast<torch::nn::Embedding*>(module.get())) {
            torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
        }
    }
}

torch::Tensor GPTTransformer::create_causal_mask(int64_t seq_len, const torch::Device& device) {
    auto mask = torch::tril(torch::ones({seq_len, seq_len}, device));
    return mask.unsqueeze(0).unsqueeze(0);
}

torch::Tensor GPTTransformer::forward(torch::Tensor x, torch::Tensor mask) {
    int64_t batch_size = x.size(0);
    int64_t seq_len = x.size(1);
    
    // Embeddings
    auto token_embeds = token_embedding_->forward(x) * std::sqrt(d_model_);
    auto pos_embeds = positional_encoding_->forward(token_embeds);
    x = dropout_->forward(pos_embeds);
    
    // Causal mask для GPT
    if (!mask.defined()) {
        mask = create_causal_mask(seq_len, x.device());
    }
    
    // Transformer blocks
    for (size_t i = 0; i < transformer_blocks->size(); ++i) {
        auto block = transformer_blocks[i]->as<TransformerBlock>();
        x = block->forward(x, mask);
    }
    
    // Output
    x = layer_norm_->forward(x);
    return lm_head_->forward(x);
}

std::string GPTTransformer::generate(SimpleTokenizer& tokenizer, const std::string& prompt,
                                      int64_t max_length, double temperature,
                                      int64_t top_k, double top_p) {
    eval();
    auto device = parameters(true)[0].device();
    
    // Токенизация промпта
    auto tokens = tokenizer.encode(prompt);
    
    // Если пусто, начинаем с BOS
    if (tokens.empty()) {
        tokens = {tokenizer.special_tokens_.at("<BOS>")};
    }
    
    auto input_ids = torch::tensor(tokens, torch::kLong).unsqueeze(0).to(device);
    std::vector<int> generated = tokens;
    
    torch::NoGradGuard no_grad;
    
    for (int64_t step = 0; step < max_length; ++step) {
        int64_t current_length = input_ids.size(1);
        
        // Обрезаем до max_seq_len если нужно
        if (current_length > max_seq_len_) {
            input_ids = input_ids.index({torch::indexing::Slice(), 
                                         torch::indexing::Slice(-max_seq_len_, None)});
            current_length = max_seq_len_;
        }
        
        // Forward pass
        auto logits = forward(input_ids);
        auto next_token_logits = logits[0][-1] / std::max(temperature, 0.1);
        
        // Top-k filtering
        if (top_k > 0) {
            int64_t k = std::min(top_k, next_token_logits.size(0));
            torch::Tensor top_k_values, top_k_indices;
            std::tie(top_k_values, top_k_indices) = torch::topk(next_token_logits, k);
            
            auto mask = torch::full_like(next_token_logits, -1e9);
            mask.scatter_(0, top_k_indices, top_k_values);
            next_token_logits = mask;
        }
        
        // Top-p (nucleus) sampling
        if (top_p < 1.0) {
            torch::Tensor sorted_probs, sorted_indices;
            std::tie(sorted_probs, sorted_indices) = torch::sort(
                torch::softmax(next_token_logits, -1), true);
            
            auto cumulative_probs = torch::cumsum(sorted_probs, -1);
            auto sorted_indices_to_remove = cumulative_probs > top_p;
            sorted_indices_to_remove.index_put_(
                {torch::indexing::Slice(1, None)}, 
                sorted_indices_to_remove.index({torch::indexing::Slice(None, -1)})
            );
            sorted_indices_to_remove.index_put_({0}, false);
            
            auto indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove);
            next_token_logits.masked_fill_(indices_to_remove, -1e9);
        }
        
        auto probs = torch::softmax(next_token_logits, -1);
        
        // Sample
        auto next_token = torch::multinomial(probs, 1);
        
        // Проверяем на конец последовательности
        int next_token_id = next_token.item<int>();
        if (next_token_id == tokenizer.eos_token_id_) {
            break;
        }
        
        generated.push_back(next_token_id);
        input_ids = torch::cat({input_ids, next_token.unsqueeze(0).unsqueeze(0)}, 1);
    }
    
    // Декодируем только сгенерированную часть
    size_t prompt_len = tokenizer.encode(prompt).size();
    std::vector<int> generated_tokens(generated.begin() + prompt_len, generated.end());
    
    return tokenizer.decode(generated_tokens);
}

std::unordered_map<std::string, int64_t> GPTTransformer::get_info() const {
    int64_t params = 0;
    for (const auto& param : parameters(true)) {
        params += param.numel();
    }
    
    return {
        {"model_id", 0}, // string, handled separately
        {"version", 0},
        {"parameters", params},
        {"vocab_size", vocab_size_},
        {"d_model", d_model_},
        {"max_seq_len", max_seq_len_}
    };
}

bool GPTTransformer::save(const std::string& path) const {
    try {
        torch::save(this, path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false;
    }
}

bool GPTTransformer::load(const std::string& path, const torch::Device& device) {
    try {
        torch::load(this, path, device);
        to(device);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

} // namespace cerebra
