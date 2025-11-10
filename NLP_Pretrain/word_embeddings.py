# ComfyNN NLP Word Embeddings Nodes
import torch
import torch.nn as nn
import numpy as np
from collections import Counter, defaultdict
import random

# 定义TENSOR类型
class TensorDataType:
    TENSOR = "TENSOR"

# ========== Word2Vec相关节点 ==========

class Word2VecSelfSupervised:
    """自监督的word2vec模型"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "corpus": ("STRING", {"multiline": True, "default": "the quick brown fox jumps over the lazy dog"}),
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
                "window_size": ("INT", {"default": 5, "min": 1, "max": 10}),
            },
            "optional": {
                "use_subsampling": ("BOOLEAN", {"default": False}),
                "subsample_threshold": ("FLOAT", {"default": 1e-5, "min": 1e-7, "max": 1e-3, "step": 1e-6}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR, "STRING")
    RETURN_NAMES = ("embeddings", "vocabulary")
    FUNCTION = "train"
    CATEGORY = "ComfyNN/NLP_Pretrain/WordEmbeddings"
    DESCRIPTION = "自监督的word2vec模型"

    def train(self, corpus, embedding_dim, window_size, use_subsampling=False, subsample_threshold=1e-5):
        # TODO: 实现完整的Word2Vec训练算法
        # 分词
        words = corpus.lower().split()
        
        # 如果启用下采样
        if use_subsampling:
            word_counts = Counter(words)
            total_words = len(words)
            subsampled_words = []
            for word in words:
                freq = word_counts[word] / total_words
                # 计算保留概率
                keep_prob = min(1.0, (subsample_threshold / freq) ** 0.5)
                if random.random() < keep_prob:
                    subsampled_words.append(word)
            words = subsampled_words
        
        # 构建词汇表
        vocab = list(set(words))
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        # TODO: 实现真正的Word2Vec训练，当前仅为初始化
        embeddings = torch.randn(vocab_size, embedding_dim) * 0.1
        
        # 简化处理，仅返回初始化的嵌入和词汇表
        vocab_str = ", ".join(vocab)
        return (embeddings, vocab_str)


class SkipGramModel:
    """跳元模型(Skip-gram Model)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "center_word": ("STRING", {"default": "learning"}),
                "context_words": ("STRING", {"default": "deep machine neural network"}),
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
            },
            "optional": {
                "use_negative_sampling": ("BOOLEAN", {"default": False}),
                "num_neg_samples": ("INT", {"default": 5, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR, TensorDataType.TENSOR)
    RETURN_NAMES = ("center_embeddings", "context_embeddings")
    FUNCTION = "compute"
    CATEGORY = "ComfyNN/NLP_Pretrain/WordEmbeddings"
    DESCRIPTION = "跳元模型(Skip-gram Model)"

    def compute(self, center_word, context_words, embedding_dim, use_negative_sampling=False, num_neg_samples=5):
        # TODO: 实现完整的Skip-gram模型训练
        # 初始化中心词和上下文词的嵌入
        center_embeddings = torch.randn(embedding_dim) * 0.1
        context_embeddings = torch.randn(embedding_dim) * 0.1
        
        # 如果启用负采样
        if use_negative_sampling:
            negative_samples = torch.randn(num_neg_samples, embedding_dim) * 0.1
            # TODO: 实现真正的负采样训练过程
            
        return (center_embeddings, context_embeddings)


class CBOWModel:
    """连续词袋模型(Continuous Bag-of-Words Model)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_word": ("STRING", {"default": "network"}),
                "context_words": ("STRING", {"default": "neural deep learning artificial"}),
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
            },
            "optional": {
                "average_context": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR, TensorDataType.TENSOR)
    RETURN_NAMES = ("target_embeddings", "context_embeddings")
    FUNCTION = "compute"
    CATEGORY = "ComfyNN/NLP_Pretrain/WordEmbeddings"
    DESCRIPTION = "连续词袋模型(Continuous Bag-of-Words Model)"

    def compute(self, target_word, context_words, embedding_dim, average_context=True):
        # TODO: 实现完整的CBOW模型训练
        # 初始化目标词和上下文词的嵌入
        target_embeddings = torch.randn(embedding_dim) * 0.1
        context_embeddings = torch.randn(embedding_dim) * 0.1
        
        # 如果需要平均上下文词向量
        if average_context:
            context_list = context_words.split()
            if len(context_list) > 1:
                # TODO: 实现上下文词向量的真正平均计算
                pass
                
        return (target_embeddings, context_embeddings)


class SubsamplingNLP:
    """适用于NLP的下采样"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "words": ("STRING", {"multiline": True, "default": "the quick brown fox jumps over the lazy dog"}),
                "threshold": ("FLOAT", {"default": 1e-5, "min": 1e-7, "max": 1e-3, "step": 1e-6}),
            },
            "optional": {
                "return_stats": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("subsampled_words", "stats")
    FUNCTION = "subsample"
    CATEGORY = "ComfyNN/NLP_Pretrain/WordEmbeddings"
    DESCRIPTION = "适用于NLP的下采样"

    def subsample(self, words, threshold, return_stats=False):
        # TODO: 优化下采样算法性能
        word_list = words.lower().split()
        word_counts = Counter(word_list)
        total_words = len(word_list)
        
        # 计算每个词的保留概率
        subsampled_words = []
        removed_count = 0
        for word in word_list:
            freq = word_counts[word] / total_words
            # 计算保留概率
            keep_prob = min(1.0, (threshold / freq) ** 0.5)
            if random.random() < keep_prob:
                subsampled_words.append(word)
            else:
                removed_count += 1
        
        stats = ""
        if return_stats:
            stats = f"Original words: {total_words}, Subsampled words: {len(subsampled_words)}, Removed: {removed_count}"
            
        return (" ".join(subsampled_words), stats)