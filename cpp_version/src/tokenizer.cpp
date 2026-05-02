/**
 * @file tokenizer.cpp
 * @brief Реализация токенизатора
 */

#include "tokenizer.hpp"
#include <iostream>
#include <cctype>

namespace cerebra {

SimpleTokenizer::SimpleTokenizer(size_t vocab_size) 
    : vocab_size_(vocab_size), eos_token_id_(3), unk_token_id_(1), next_id_(0) {
    
    // Инициализация специальных токенов
    build_vocab_from_special();
}

void SimpleTokenizer::build_vocab_from_special() {
    special_tokens_["<PAD>"] = 0;
    special_tokens_["<UNK>"] = 1;
    special_tokens_["<BOS>"] = 2;
    special_tokens_["<EOS>"] = 3;
    special_tokens_["<SEP>"] = 4;
    
    word_to_id_ = special_tokens_;
    
    for (const auto& pair : special_tokens_) {
        id_to_word_[pair.second] = pair.first;
    }
    
    next_id_ = special_tokens_.size();
    eos_token_id_ = special_tokens_["<EOS>"];
    unk_token_id_ = special_tokens_["<UNK>"];
}

std::string SimpleTokenizer::normalize_text(const std::string& text) const {
    std::string result;
    result.reserve(text.size());
    
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!result.empty() && !std::isspace(static_cast<unsigned char>(result.back()))) {
                result += ' ';
            }
        } else {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    
    // Trim
    size_t start = result.find_first_not_of(" \t\n\r");
    size_t end = result.find_last_not_of(" \t\n\r");
    
    if (start == std::string::npos) return "";
    return result.substr(start, end - start + 1);
}

std::vector<std::string> SimpleTokenizer::tokenize_text(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string normalized = normalize_text(text);
    
    if (normalized.empty()) return tokens;
    
    // Простой regex для слов и знаков препинания
    std::regex word_regex(R"(\b\w+|[^\w\s])");
    auto words_begin = std::sregex_iterator(normalized.begin(), normalized.end(), word_regex);
    auto words_end = std::sregex_iterator();
    
    for (auto it = words_begin; it != words_end; ++it) {
        tokens.push_back(it->str());
    }
    
    return tokens;
}

size_t SimpleTokenizer::build_vocab(const std::vector<std::string>& texts) {
    std::unordered_map<std::string, int> word_counts;
    
    // Подсчет частоты слов
    for (const auto& text : texts) {
        auto tokens = tokenize_text(text);
        for (const auto& token : tokens) {
            word_counts[token]++;
        }
    }
    
    // Сортировка по частоте
    std::vector<std::pair<std::string, int>> sorted_words(word_counts.begin(), word_counts.end());
    std::sort(sorted_words.begin(), sorted_words.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Добавление самых частых слов в словарь
    size_t max_vocab = vocab_size_ - word_to_id_.size();
    for (const auto& pair : sorted_words) {
        if (word_to_id_.size() >= vocab_size_) break;
        if (word_to_id_.find(pair.first) == word_to_id_.end()) {
            word_to_id_[pair.first] = next_id_;
            id_to_word_[next_id_] = pair.first;
            next_id_++;
        }
    }
    
    return word_to_id_.size();
}

std::vector<int> SimpleTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    auto tokens = tokenize_text(text);
    
    for (const auto& token : tokens) {
        auto it = word_to_id_.find(token);
        if (it != word_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unk_token_id_);
        }
    }
    
    return ids;
}

std::string SimpleTokenizer::decode(const std::vector<int>& ids) const {
    std::vector<std::string> words;
    
    for (int id : ids) {
        auto it = id_to_word_.find(id);
        if (it != id_to_word_.end()) {
            const std::string& word = it->second;
            // Пропускаем специальные токены кроме тех, что нужны
            if (special_tokens_.find(word) == special_tokens_.end()) {
                words.push_back(word);
            }
        } else if (id != unk_token_id_) {
            words.push_back("<UNK>");
        }
    }
    
    // Сборка текста
    std::string text;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            // Проверка на знаки препинания
            bool is_punct = !words[i].empty() && 
                           !std::isalpha(static_cast<unsigned char>(words[i][0]));
            if (!is_punct) {
                text += " ";
            }
        }
        text += words[i];
    }
    
    // Удаление лишних пробелов вокруг знаков препинания
    std::regex punct_space_regex(R"(\s+([^\w\s]))");
    text = std::regex_replace(text, punct_space_regex, "$1");
    std::regex space_punct_regex(R"(([^\w\s])\s+)");
    text = std::regex_replace(text, space_punct_regex, "$1 ");
    
    return text;
}

bool SimpleTokenizer::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    file << "{\n";
    file << "  \"vocab_size\": " << vocab_size_ << ",\n";
    
    // word_to_id
    file << "  \"word_to_id\": {\n";
    bool first = true;
    for (const auto& pair : word_to_id_) {
        if (!first) file << ",\n";
        first = false;
        file << "    \"" << pair.first << "\": " << pair.second;
    }
    file << "\n  },\n";
    
    // id_to_word
    file << "  \"id_to_word\": {\n";
    first = true;
    for (const auto& pair : id_to_word_) {
        if (!first) file << ",\n";
        first = false;
        file << "    \"" << pair.first << "\": \"" << pair.second << "\"";
    }
    file << "\n  },\n";
    
    // special_tokens
    file << "  \"special_tokens\": {\n";
    first = true;
    for (const auto& pair : special_tokens_) {
        if (!first) file << ",\n";
        first = false;
        file << "    \"" << pair.first << "\": " << pair.second;
    }
    file << "\n  }\n";
    
    file << "}\n";
    
    return true;
}

bool SimpleTokenizer::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    // Простая парсинг JSON (для продакшена лучше использовать nlohmann/json)
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // Здесь должна быть полноценная парсинг JSON
    // Для краткости используем упрощенный подход
    
    // В реальной реализации нужно использовать JSON библиотеку
    std::cerr << "Warning: Full JSON parsing not implemented yet" << std::endl;
    
    return true;
}

size_t SimpleTokenizer::get_vocab_size() const {
    return word_to_id_.size();
}

} // namespace cerebra
