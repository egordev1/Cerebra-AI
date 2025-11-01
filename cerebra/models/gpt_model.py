"""
Трансформер модель для генерации текста (GPT-like архитектура)
Файл: gpt_model.py - GPT Transformer модель (главный мозг ИИ)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import logging
from collections import Counter

try:
    from cerebra.logger_config import logger
except ImportError:
    logger = logging.getLogger('cerebra')


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention механизм"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attention_output)


class TransformerBlock(nn.Module):
    """Один блок трансформера"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention с residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward с residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class GPTTransformer(nn.Module):
    """GPT-like трансформер модель для генерации текста"""
    def __init__(self, vocab_size=10000, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.model_id = "Synthesis-L1-GPT"
        self.version = "2.0.0"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Инициализация весов
        self.apply(self._init_weights)
            
        logger.info(f"🎯 Создана {self.model_id} v{self.version}")
        logger.info(f"   Параметров: {sum(p.numel() for p in self.parameters()):,}")
        print(f"🎯 Создана {self.model_id} v{self.version}")
    
    def _init_weights(self, module):
        """Инициализация весов"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len, device):
        """Создание каузальной маски для GPT"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, x, mask=None):
        """
        Forward pass
        x: [batch_size, seq_len]
        """
        batch_size, seq_len = x.size()
        
        # Embeddings
        token_embeds = self.token_embedding(x) * math.sqrt(self.d_model)
        pos_embeds = self.positional_encoding(token_embeds.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(pos_embeds)
        
        # Causal mask для GPT
        if mask is None:
            mask = self.create_causal_mask(seq_len, x.device)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Output
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
        """
        Генерация текста на основе промпта
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Токенизация промпта
        tokens = tokenizer.encode(prompt)
        
        # Если пусто, начинаем с BOS
        if not tokens:
            tokens = [tokenizer.special_tokens['<BOS>']]
        
        input_ids = torch.tensor([tokens], device=device)
        generated = tokens.copy()
        
        with torch.no_grad():
            for step in range(max_length):
                # Обрезаем до max_seq_len если нужно
                current_length = input_ids.size(1)
                if current_length > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]
                    current_length = self.max_seq_len
                
                # Forward pass
                logits = self(input_ids)
                next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                
                # Top-k filtering
                if top_k > 0:
                    top_k_value = min(top_k, next_token_logits.size(-1))
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k_value)
                    top_k_mask = torch.zeros_like(next_token_logits).fill_(-float('inf'))
                    top_k_mask.scatter_(0, top_k_indices, top_k_logits)
                    next_token_logits = top_k_mask
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Удаляем токены с кумулятивной вероятностью > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Избегаем специальных токенов кроме EOS
                if tokenizer.special_tokens['<PAD>'] < len(probs):
                    probs[tokenizer.special_tokens['<PAD>']] = 0
                if tokenizer.special_tokens['<UNK>'] < len(probs):
                    probs[tokenizer.special_tokens['<UNK>']] *= 0.5  # Снижаем вероятность UNK
                
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Проверяем на конец последовательности
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                next_token_id = next_token.item()
                generated.append(next_token_id)
                
                # Правильное добавление токена
                # next_token имеет размер [1] после multinomial
                # input_ids имеет размер [1, seq_len]
                # Нужно преобразовать next_token в [1, 1]
                if next_token.dim() == 0:
                    # Если скаляр, делаем два unsqueeze
                    next_token_tensor = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
                elif next_token.dim() == 1:
                    # Если уже одномерный [1], делаем один unsqueeze
                    next_token_tensor = next_token.unsqueeze(0)  # [1, 1]
                else:
                    # Если уже [1, 1], оставляем как есть
                    next_token_tensor = next_token
                
                # Проверяем размерности перед конкатенацией
                if input_ids.dim() == 2 and next_token_tensor.dim() == 2:
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=1)  # [1, seq_len+1]
                else:
                    logger.error(f"Несовместимые размерности: input_ids={input_ids.shape}, next_token={next_token_tensor.shape}")
                    break
        
        # Декодируем только сгенерированную часть (без исходного промпта)
        # Вычисляем длину исходного промпта
        prompt_tokens = tokenizer.encode(prompt)
        if len(generated) > len(prompt_tokens):
            generated_tokens = generated[len(prompt_tokens):]
        else:
            generated_tokens = generated
        
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text.strip()
    
    def get_info(self):
        return {
            'model_id': self.model_id,
            'version': self.version,
            'parameters': sum(p.numel() for p in self.parameters()),
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len
        }

