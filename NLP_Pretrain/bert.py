# ComfyNN NLP Pretrain BERT Nodes
# Based on d2l-zh implementation (https://github.com/d2l-ai/d2l-zh)
# Thank you d2l-ai team for the excellent educational resource

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", self.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

    class EncoderBlock(nn.Module):
        """Transformer编码器块"""
        def __init__(self, key_size, query_size, value_size, num_hiddens,
                     norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                     dropout, use_bias=False, **kwargs):
            super().__init__(**kwargs)
            self.attention = self.MultiHeadAttention(
                key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
            self.addnorm1 = self.AddNorm(norm_shape, dropout)
            self.ffn = self.PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
            self.addnorm2 = self.AddNorm(norm_shape, dropout)

        def forward(self, X, valid_lens):
            Y, _ = self.attention(X, X, X, valid_lens)
            X = self.addnorm1(X, Y)
            Y = self.ffn(X)
            X = self.addnorm2(X, Y)
            return X

        class MultiHeadAttention(nn.Module):
            """多头注意力"""
            def __init__(self, key_size, query_size, value_size, num_hiddens,
                         num_heads, dropout, bias=False, **kwargs):
                super().__init__(**kwargs)
                self.num_heads = num_heads
                self.attention = self.DotProductAttention(dropout)
                self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
                self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
                self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
                self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

            def forward(self, queries, keys, values, valid_lens):
                queries = self.W_q(queries)
                keys = self.W_k(keys)
                values = self.W_v(values)
                
                queries = self.transpose_qkv(queries)
                keys = self.transpose_qkv(keys)
                values = self.transpose_qkv(values)
                
                if valid_lens is not None:
                    valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
                
                output, attention_weights = self.attention(queries, keys, values, valid_lens)
                output = self.transpose_output(output)
                return self.W_o(output), attention_weights

            def transpose_qkv(self, X):
                """变换形状以实现多头注意力"""
                X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
                X = X.permute(0, 2, 1, 3)
                return X.reshape(-1, X.shape[2], X.shape[3])

            def transpose_output(self, X):
                """还原形状"""
                X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
                X = X.permute(0, 2, 1, 3)
                return X.reshape(X.shape[0], X.shape[1], -1)

            class DotProductAttention(nn.Module):
                """点积注意力"""
                def __init__(self, dropout, **kwargs):
                    super().__init__(**kwargs)
                    self.dropout = nn.Dropout(dropout)

                def forward(self, queries, keys, values, valid_lens=None):
                    d = queries.shape[-1]
                    scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
                    
                    if valid_lens is not None:
                        scores = self.mask_softmax(scores, valid_lens)
                    else:
                        self.attention_weights = F.softmax(scores, dim=-1)
                    
                    return torch.bmm(self.dropout(self.attention_weights), values), self.attention_weights

                def mask_softmax(self, X, valid_lens):
                    """遮蔽softmax"""
                    if valid_lens is None:
                        return F.softmax(X, dim=-1)
                    else:
                        shape = X.shape
                        if valid_lens.dim() == 1:
                            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
                        else:
                            valid_lens = valid_lens.reshape(-1)
                        X = X.reshape((-1, shape[-1]))
                        max_len = X.shape[1]
                        batch_size = X.shape[0]
                        valid_lens = valid_lens.reshape((batch_size, 1))
                        mask = torch.arange(max_len, dtype=torch.float32, device=X.device).reshape((1, max_len)).expand(batch_size, max_len) >= valid_lens
                        X = X.masked_fill_(mask, -1e6)
                        self.attention_weights = F.softmax(X.reshape(shape), dim=-1)
                        return self.attention_weights

        class AddNorm(nn.Module):
            """残差连接后进行层规范化"""
            def __init__(self, normalized_shape, dropout, **kwargs):
                super().__init__(**kwargs)
                self.dropout = nn.Dropout(dropout)
                self.ln = nn.LayerNorm(normalized_shape)

            def forward(self, X, Y):
                return self.ln(X + self.dropout(Y))

        class PositionWiseFFN(nn.Module):
            """基于位置的前馈网络"""
            def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
                super().__init__(**kwargs)
                self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
                self.relu = nn.ReLU()
                self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

            def forward(self, X):
                return self.dense2(self.relu(self.dense1(X)))


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class BERTModel:
    """BERT模型节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vocab_size": ("INT", {"default": 10000, "min": 1000, "max": 100000}),
                "num_hiddens": ("INT", {"default": 768, "min": 128, "max": 2048}),
                "ffn_num_hiddens": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "num_heads": ("INT", {"default": 4, "min": 1, "max": 16}),
                "num_layers": ("INT", {"default": 2, "min": 1, "max": 24}),
                "dropout": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.9, "step": 0.05}),
            },
            "optional": {
                "max_len": ("INT", {"default": 1000, "min": 100, "max": 5000}),
            }
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("bert_model",)
    FUNCTION = "create_bert"
    CATEGORY = "ComfyNN/NLP_Pretrain/BERT"
    DESCRIPTION = "Create BERT model based on d2l-zh implementation"

    def create_bert(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, 
                    num_layers, dropout, max_len=1000):
        norm_shape = [num_hiddens]
        ffn_num_input = num_hiddens
        
        bert_encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                                   ffn_num_hiddens, num_heads, num_layers, dropout, max_len)
        
        return (bert_encoder,)


class BERTMaskedLanguageModel:
    """BERT掩码语言模型节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bert_model": ("CUSTOM",),
                "mlm_weights": ("TENSOR",),
                "mlm_bias": ("TENSOR",),
                "tokens": ("STRING", {"multiline": True, "default": "this is a masked language model example"}),
                "masked_positions": ("STRING", {"default": "1,3,5"}),  # 以逗号分隔的位置
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("predictions", "mlm_info")
    FUNCTION = "predict_masked_tokens"
    CATEGORY = "ComfyNN/NLP_Pretrain/BERT"
    DESCRIPTION = "Predict masked tokens using BERT model"

    def predict_masked_tokens(self, bert_model, mlm_weights, mlm_bias, tokens, masked_positions):
        # 解析输入
        token_list = tokens.strip().split()
        masked_positions_list = [int(pos) for pos in masked_positions.split(",") if pos.strip()]
        
        # 构建输入
        input_tokens, segments = get_tokens_and_segments(token_list)
        
        # 转换为索引（简化处理，实际应该使用词汇表）
        # 这里我们只是演示结构，实际实现需要完整的词汇表处理
        vocab_size = bert_model.token_embedding.num_embeddings
        token_indices = torch.randint(0, vocab_size, (1, len(input_tokens)))
        segment_indices = torch.tensor([segments])
        valid_lens = torch.tensor([len(input_tokens)])
        
        # BERT编码
        encoded_X = bert_model(token_indices, segment_indices, valid_lens)
        
        # 获取遮蔽位置的词元表示
        masked_positions_tensor = torch.tensor(masked_positions_list)
        masked_X = encoded_X[:, masked_positions_tensor, :]
        
        # 通过MLM头预测
        predictions = torch.matmul(masked_X, mlm_weights.transpose(1, 0)) + mlm_bias
        
        # 生成信息
        mlm_info = f"BERT MLM Prediction\n"
        mlm_info += f"Input tokens: {len(input_tokens)}\n"
        mlm_info += f"Masked positions: {len(masked_positions_list)}\n"
        mlm_info += f"Vocabulary size: {vocab_size}\n"
        mlm_info += f"Predictions shape: {list(predictions.shape)}"
        
        return (predictions, mlm_info)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "BERTModel": BERTModel,
    "BERTMaskedLanguageModel": BERTMaskedLanguageModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BERTModel": "BERT Model 🐱",
    "BERTMaskedLanguageModel": "BERT Masked Language Model 🐱",
}