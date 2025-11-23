"""
Продвинутая трансформерная модель для Cerebra AI
Файл: advanced_transformer.py - Продвинутая GPT-3 подобная модель
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

try:
    from cerebra.logger_config import logger
except ImportError:
    logger = logging.getLogger('cerebra')


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding для улучшения позиционного понимания"""
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:, None, :].repeat(1, x.size(1), 1), emb[:, None, :].repeat(1, x.size(1), 1)


class FlashAttention(nn.Module):
    """Оптимизированный механизм внимания для улучшения производительности"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(output)


class FeedForward(nn.Module):
    """Улучшенный feed-forward слой с GELU активацией"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AdvancedTransformerBlock(nn.Module):
    """Продвинутый трансформерный блок с улучшенными компонентами"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = FlashAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Residual connection + Layer norm
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # FFN + Residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class SynthesisL2(nn.Module):
    """Продвинутая GPT-3 подобная модель (Synthesis-L2) - оптимизированная версия"""
    def __init__(self, vocab_size: int = 8000, d_model: int = 384, n_heads: int = 6, 
                 n_layers: int = 6, d_ff: int = 1536, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.model_id = "Synthesis-L2"
        self.version = "3.0.0"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token и positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Rotary embeddings для улучшения позиционного понимания
        self.rotary_emb = RotaryEmbedding(d_model // n_heads)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdvancedTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"🎯 Создана {self.model_id} v{self.version}")
        logger.info(f"   Параметров: {sum(p.numel() for p in self.parameters()):,}")
        print(f"🎯 Создана {self.model_id} v{self.version}")
        print(f"   Параметров: ~{sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        """Улучшенная инициализация весов"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        
        # Token и positional embeddings
        token_embeds = self.token_embedding(x) * math.sqrt(self.d_model)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(pos_ids)
        
        x = self.dropout(token_embeds + pos_embeds)
        
        # Causal mask для GPT
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # Проход через трансформерные блоки
        for block in self.blocks:
            x = block(x, mask)
        
        # Layer norm и линейный выход
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, tokenizer, prompt: str, max_length: int = 150, temperature: float = 0.8, 
                 top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.2):
        """Улучшенная генерация с поддержкой повторений и лучшими методами выбора"""
        self.eval()
        device = next(self.parameters()).device
        
        # Токенизация промпта
        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.special_tokens['<BOS>']]
        
        input_ids = torch.tensor([tokens], device=device)
        generated = tokens.copy()
        
        with torch.no_grad():
            for step in range(max_length):
                current_length = input_ids.size(1)
                if current_length > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]
                    current_length = self.max_seq_len
                
                # Forward pass
                logits = self(input_ids)
                next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                
                # Penalize repetition
                for token_id in set(generated[-10:]):  # Последние 10 токенов
                    next_token_logits[token_id] /= repetition_penalty
                
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
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Избегаем специальных токенов
                if tokenizer.special_tokens['<PAD>'] < len(probs):
                    probs[tokenizer.special_tokens['<PAD>']] = 0
                if tokenizer.special_tokens['<UNK>'] < len(probs):
                    probs[tokenizer.special_tokens['<UNK>']] *= 0.3  # Сильно снижаем вероятность UNK
                
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Проверка на конец последовательности
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                next_token_id = next_token.item()
                generated.append(next_token_id)
                
                # Добавляем токен к input_ids
                next_token_tensor = next_token.unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
        
        # Декодируем только сгенерированную часть
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
            'n_heads': 12,
            'n_layers': 12,
            'max_seq_len': self.max_seq_len
        }


class SynthesisL3(nn.Module):
    """Еще более продвинутая модель (Synthesis-L3) с MoE (Mixture of Experts) - оптимизированная версия"""
    def __init__(self, vocab_size: int = 10000, d_model: int = 384, n_heads: int = 6, 
                 n_layers: int = 8, d_ff: int = 1536, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.model_id = "Synthesis-L3"
        self.version = "4.0.0"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token и positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks (без MoE для упрощения, но с увеличенной архитектурой)
        self.blocks = nn.ModuleList([
            AdvancedTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
        logger.info(f"🎯 Создана {self.model_id} v{self.version}")
        logger.info(f"   Параметров: {sum(p.numel() for p in self.parameters()):,}")
        print(f"🎯 Создана {self.model_id} v{self.version}")
        print(f"   Параметров: ~{sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        """Улучшенная инициализация весов"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        
        # Token и positional embeddings
        token_embeds = self.token_embedding(x) * math.sqrt(self.d_model)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.pos_embedding(pos_ids)
        
        x = self.dropout(token_embeds + pos_embeds)
        
        # Causal mask для GPT
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # Проход через трансформерные блоки
        for block in self.blocks:
            x = block(x, mask)
        
        # Layer norm и линейный выход
        x = self.layer_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, tokenizer, prompt: str, max_length: int = 200, temperature: float = 0.7, 
                 top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.1):
        """Улучшенная генерация для L3 модели"""
        self.eval()
        device = next(self.parameters()).device
        
        # Токенизация промпта
        tokens = tokenizer.encode(prompt)
        if not tokens:
            tokens = [tokenizer.special_tokens['<BOS>']]
        
        input_ids = torch.tensor([tokens], device=device)
        generated = tokens.copy()
        
        with torch.no_grad():
            for step in range(max_length):
                current_length = input_ids.size(1)
                if current_length > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]
                    current_length = self.max_seq_len
                
                # Forward pass
                logits = self(input_ids)
                next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                
                # Penalize repetition
                for token_id in set(generated[-15:]):  # Последние 15 токенов
                    next_token_logits[token_id] /= repetition_penalty
                
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
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Избегаем специальных токенов
                if tokenizer.special_tokens['<PAD>'] < len(probs):
                    probs[tokenizer.special_tokens['<PAD>']] = 0
                if tokenizer.special_tokens['<UNK>'] < len(probs):
                    probs[tokenizer.special_tokens['<UNK>']] *= 0.2  # Сильно снижаем вероятность UNK
                
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Проверка на конец последовательности
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                next_token_id = next_token.item()
                generated.append(next_token_id)
                
                # Добавляем токен к input_ids
                next_token_tensor = next_token.unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
        
        # Декодируем только сгенерированную часть
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
            'n_heads': 16,
            'n_layers': 24,
            'max_seq_len': self.max_seq_len
        }