/**
 * @file tokenizer.hpp
 * @brief Токенизатор для русского языка (BPE-like)
 * Преобразование текста в токены и обратно, построение словаря
 */

#ifndef CEREBRA_TOKENIZER_HPP
#define CEREBRA_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace cerebra {

class SimpleTokenizer {
public:
    SimpleTokenizer(size_t vocab_size = 10000);
    
    // Построение словаря из текстов
    size_t build_vocab(const std::vector<std::string>& texts);
    
    // Кодирование текста в последовательность токенов
    std::vector<int> encode(const std::string& text) const;
    
    // Декодирование последовательности токенов в текст
    std::string decode(const std::vector<int>& ids) const;
    
    // Сохранение токенизатора в файл
    bool save(const std::string& path) const;
    
    // Загрузка токенизатора из файла
    bool load(const std::string& path);
    
    // Получить размер словаря
    size_t get_vocab_size() const;

private:
    size_t vocab_size_;
    std::unordered_map<std::string, int> word_to_id_;
    std::unordered_map<int, std::string> id_to_word_;
    std::unordered_map<std::string, int> special_tokens_;
    int eos_token_id_;
    int unk_token_id_;
    int next_id_;
    
    // Инициализация словаря специальными токенами
    void build_vocab_from_special();
    
    // Разбиение текста на токены
    std::vector<std::string> tokenize_text(const std::string& text) const;
    
    // Нормализация текста
    std::string normalize_text(const std::string& text) const;
};

} // namespace cerebra

#endif // CEREBRA_TOKENIZER_HPP
