# ComfyNN NLP Pretrain Test Data Generator
import torch
import random
import numpy as np

class TensorDataType:
    TENSOR = "TENSOR"

class NLPTestDataGenerator:
    """生成用于NLP预训练模块测试的随机数据"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task_type": (["word2vec", "glove", "bert"], {"default": "word2vec"}),
                "vocab_size": ("INT", {"default": 1000, "min": 100, "max": 10000}),
                "sequence_length": ("INT", {"default": 128, "min": 10, "max": 512}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
                "window_size": ("INT", {"default": 5, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR, TensorDataType.TENSOR, "STRING")
    RETURN_NAMES = ("input_data", "target_data", "sample_text")
    FUNCTION = "generate"
    CATEGORY = "ComfyNN/NLP_Pretrain/Testing"
    DESCRIPTION = "生成用于NLP预训练模块测试的随机数据"

    def generate(self, task_type, vocab_size, sequence_length, batch_size, seed, 
                 embedding_dim=100, window_size=5):
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if task_type == "word2vec":
            # 生成Word2Vec训练数据 (中心词, 上下文词)
            center_words = torch.randint(0, vocab_size, (batch_size,))
            context_words = torch.randint(0, vocab_size, (batch_size,))
            sample_text = "Sample text for Word2Vec training"
            return (center_words, context_words, sample_text)
            
        elif task_type == "glove":
            # 生成GloVe训练数据 (共现矩阵)
            coocurrence_matrix = torch.randn(vocab_size, vocab_size)
            target_data = torch.randn(vocab_size, embedding_dim)
            sample_text = "Sample text for GloVe training"
            return (coocurrence_matrix, target_data, sample_text)
            
        elif task_type == "bert":
            # 生成BERT训练数据 (输入序列)
            input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
            attention_mask = torch.ones(batch_size, sequence_length)
            sample_text = "Sample text for BERT training [MASK] token"
            return (input_ids, attention_mask, sample_text)