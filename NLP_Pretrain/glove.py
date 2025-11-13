# ComfyNN NLP Pretrain GloVe Node
# Based on d2l-zh implementation (https://github.com/d2l-ai/d2l-zh)
# Thank you d2l-ai team for the excellent educational resource

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import math

class GloVeNode:
    """GloVe (Global Vectors) word embedding node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_corpus": ("STRING", {"multiline": True, "default": "the man loves his son\nthe cat sat on the mat"}),
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
                "context_size": ("INT", {"default": 2, "min": 1, "max": 5}),
            },
            "optional": {
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "num_epochs": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "x_max": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
                "alpha": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("word_embeddings", "context_embeddings", "training_info")
    FUNCTION = "train_glove"
    CATEGORY = "ComfyNN/NLP_Pretrain/GloVe"
    DESCRIPTION = "Train GloVe embeddings using global corpus statistics"

    def train_glove(self, text_corpus, embedding_dim, context_size, learning_rate=0.01, 
                    num_epochs=100, x_max=100.0, alpha=0.75):
        # Preprocess text
        sentences = text_corpus.strip().split('\n')
        tokenized_sentences = [sentence.split() for sentence in sentences]
        
        # Build vocabulary
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        vocab = list(set(all_words))
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_word = {i: word for i, word in enumerate(vocab)}
        
        # Generate co-occurrence statistics
        cooccur = self._build_cooccur_matrix(tokenized_sentences, word_to_idx, context_size)
        
        # Initialize embeddings
        center_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        context_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        center_bias = torch.randn(vocab_size, requires_grad=True)
        context_bias = torch.randn(vocab_size, requires_grad=True)
        
        # Training
        optimizer = torch.optim.SGD([center_embeddings, context_embeddings, center_bias, context_bias], 
                                  lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            count = 0
            
            for i in range(vocab_size):
                for j in range(vocab_size):
                    if cooccur[i][j] > 0:
                        # Weight function
                        weight = self._weight_function(cooccur[i][j], x_max, alpha)
                        
                        # Forward pass
                        dot_product = torch.dot(center_embeddings[i], context_embeddings[j])
                        prediction = dot_product + center_bias[i] + context_bias[j]
                        diff = prediction - math.log(cooccur[i][j])
                        loss = weight * (diff ** 2)
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        count += 1
            
            if epoch % 20 == 0 and count > 0:
                avg_loss = total_loss / count
                print(f"GloVe Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        # Combine center and context embeddings
        final_embeddings = center_embeddings + context_embeddings
        
        # Generate training info
        training_info = f"GloVe Training Completed\n"
        training_info += f"Vocabulary size: {vocab_size}\n"
        training_info += f"Embedding dimension: {embedding_dim}\n"
        training_info += f"Context size: {context_size}\n"
        training_info += f"Final loss: {total_loss/count if count > 0 else 0:.4f}\n"
        training_info += f"Sample words: {', '.join(vocab[:5])}"
        
        return (final_embeddings.detach(), context_embeddings.detach(), training_info)

    def _build_cooccur_matrix(self, tokenized_sentences, word_to_idx, context_size):
        """Build co-occurrence matrix from tokenized sentences"""
        vocab_size = len(word_to_idx)
        cooccur = np.zeros((vocab_size, vocab_size))
        
        for sentence in tokenized_sentences:
            for i, center_word in enumerate(sentence):
                center_idx = word_to_idx[center_word]
                # Get context words within window
                start = max(0, i - context_size)
                end = min(len(sentence), i + context_size + 1)
                for j in range(start, end):
                    if j != i:
                        context_idx = word_to_idx[sentence[j]]
                        # Distance weighting (closer words have higher weights)
                        distance = abs(i - j)
                        cooccur[center_idx][context_idx] += 1.0 / distance
        
        return cooccur

    def _weight_function(self, x_ij, x_max, alpha):
        """Weight function for GloVe loss"""
        if x_ij == 0:
            return 0
        elif x_ij < x_max:
            return (x_ij / x_max) ** alpha
        else:
            return 1.0

# Node mappings
NODE_CLASS_MAPPINGS = {
    "GloVeNode": GloVeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GloVeNode": "GloVe Embeddings 🐱",
}