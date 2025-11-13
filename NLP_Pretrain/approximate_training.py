# ComfyNN NLP Pretrain Approximate Training Node
# Based on d2l-zh implementation (https://github.com/d2l-ai/d2l-zh)
# Thank you d2l-ai team for the excellent educational resource

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import random
import math

class NegativeSamplingNode:
    """Negative Sampling node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_corpus": ("STRING", {"multiline": True, "default": "the man loves his son\nthe cat sat on the mat"}),
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
                "context_size": ("INT", {"default": 2, "min": 1, "max": 5}),
            },
            "optional": {
                "num_negatives": ("INT", {"default": 5, "min": 1, "max": 20}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "num_epochs": ("INT", {"default": 100, "min": 10, "max": 1000}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("center_embeddings", "context_embeddings", "training_info")
    FUNCTION = "train_negative_sampling"
    CATEGORY = "ComfyNN/NLP_Pretrain/ApproximateTraining"
    DESCRIPTION = "Train word embeddings using negative sampling"

    def train_negative_sampling(self, text_corpus, embedding_dim, context_size, 
                               num_negatives=5, learning_rate=0.01, num_epochs=100):
        # Preprocess text
        sentences = text_corpus.strip().split('\n')
        tokenized_sentences = [sentence.split() for sentence in sentences]
        
        # Build vocabulary
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        vocab = list(set(all_words))
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_word = {i: word for i, word in enumerate(vocab)}
        
        # Calculate word frequencies for negative sampling
        word_counts = Counter(all_words)
        total_words = len(all_words)
        word_freq = {word: count/total_words for word, count in word_counts.items()}
        
        # Generate training data
        center_words = []
        context_words = []
        labels = []  # 1 for positive, 0 for negative
        
        for sentence in tokenized_sentences:
            for i, center_word in enumerate(sentence):
                center_idx = word_to_idx[center_word]
                # Get context words within window
                start = max(0, i - context_size)
                end = min(len(sentence), i + context_size + 1)
                for j in range(start, end):
                    if j != i:
                        context_idx = word_to_idx[sentence[j]]
                        # Positive sample
                        center_words.append(center_idx)
                        context_words.append(context_idx)
                        labels.append(1)
                        
                        # Negative samples
                        for _ in range(num_negatives):
                            # Sample negative word based on frequency
                            neg_word = self._sample_negative(word_freq, vocab, word_to_idx, 
                                                           [center_word, sentence[j]])
                            neg_idx = word_to_idx[neg_word]
                            center_words.append(center_idx)
                            context_words.append(neg_idx)
                            labels.append(0)
        
        # Convert to tensors
        center_words = torch.tensor(center_words, dtype=torch.long)
        context_words = torch.tensor(context_words, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float)
        
        # Initialize embeddings
        center_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        context_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        
        # Training
        optimizer = torch.optim.SGD([center_embeddings, context_embeddings], lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            # Process in batches to avoid memory issues
            batch_size = min(1000, len(center_words))
            num_batches = (len(center_words) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(center_words))
                
                batch_center = center_words[start_idx:end_idx]
                batch_context = context_words[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                
                # Forward pass
                center_embeds = center_embeddings[batch_center]  # [B, embedding_dim]
                context_embeds = context_embeddings[batch_context]  # [B, embedding_dim]
                
                # Compute scores
                scores = torch.sum(center_embeds * context_embeds, dim=1)  # [B]
                predictions = torch.sigmoid(scores)  # [B]
                
                # Compute loss
                loss = F.binary_cross_entropy(predictions, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / num_batches
                print(f"Negative Sampling Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        # Generate training info
        training_info = f"Negative Sampling Training Completed\n"
        training_info += f"Vocabulary size: {vocab_size}\n"
        training_info += f"Embedding dimension: {embedding_dim}\n"
        training_info += f"Context size: {context_size}\n"
        training_info += f"Negative samples per positive: {num_negatives}\n"
        training_info += f"Final loss: {total_loss/num_batches:.4f}\n"
        training_info += f"Sample words: {', '.join(vocab[:5])}"
        
        return (center_embeddings.detach(), context_embeddings.detach(), training_info)

    def _sample_negative(self, word_freq, vocab, word_to_idx, exclude_words):
        """Sample a negative word based on frequency distribution"""
        # Create a list of candidate words (excluding the positive ones)
        candidates = [word for word in vocab if word not in exclude_words]
        if not candidates:
            candidates = vocab
            
        # Sample based on frequency
        words, probs = zip(*[(word, word_freq[word]) for word in candidates])
        probs = np.array(probs)
        probs = probs / np.sum(probs)  # Normalize
        return np.random.choice(words, p=probs)

class HierarchicalSoftmaxNode:
    """Hierarchical Softmax node based on d2l-zh implementation"""
    
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
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("center_embeddings", "context_embeddings", "training_info")
    FUNCTION = "train_hierarchical_softmax"
    CATEGORY = "ComfyNN/NLP_Pretrain/ApproximateTraining"
    DESCRIPTION = "Train word embeddings using hierarchical softmax"

    def train_hierarchical_softmax(self, text_corpus, embedding_dim, context_size, 
                                  learning_rate=0.01, num_epochs=100):
        # Preprocess text
        sentences = text_corpus.strip().split('\n')
        tokenized_sentences = [sentence.split() for sentence in sentences]
        
        # Build vocabulary
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        vocab = list(set(all_words))
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_word = {i: word for i, word in enumerate(vocab)}
        
        # For hierarchical softmax, we would build a binary tree
        # For simplicity in this implementation, we'll use a simplified approach
        # that approximates the concept without building the full tree
        
        # Generate training data
        center_words = []
        context_words = []
        
        for sentence in tokenized_sentences:
            for i, center_word in enumerate(sentence):
                center_idx = word_to_idx[center_word]
                # Get context words within window
                start = max(0, i - context_size)
                end = min(len(sentence), i + context_size + 1)
                for j in range(start, end):
                    if j != i:
                        context_idx = word_to_idx[sentence[j]]
                        center_words.append(center_idx)
                        context_words.append(context_idx)
        
        # Convert to tensors
        center_words = torch.tensor(center_words, dtype=torch.long)
        context_words = torch.tensor(context_words, dtype=torch.long)
        
        # Initialize embeddings
        center_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        # For hierarchical softmax, we would have node embeddings, but for simplicity
        # we'll use context embeddings
        context_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        
        # Training with simplified hierarchical softmax approach
        optimizer = torch.optim.SGD([center_embeddings, context_embeddings], lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            # Process in batches
            batch_size = min(1000, len(center_words))
            num_batches = (len(center_words) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(center_words))
                
                batch_center = center_words[start_idx:end_idx]
                batch_context = context_words[start_idx:end_idx]
                
                # Forward pass
                center_embeds = center_embeddings[batch_center]  # [B, embedding_dim]
                context_embeds = context_embeddings[batch_context]  # [B, embedding_dim]
                
                # Compute scores (simplified hierarchical softmax)
                # In a real implementation, this would involve tree paths
                scores = torch.sum(center_embeds * context_embeds, dim=1)  # [B]
                # Using log-sum-exp for numerical stability
                log_probs = scores - torch.logsumexp(scores.unsqueeze(1).repeat(1, vocab_size), dim=1)
                loss = -torch.mean(log_probs)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / num_batches
                print(f"Hierarchical Softmax Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        # Generate training info
        training_info = f"Hierarchical Softmax Training Completed\n"
        training_info += f"Vocabulary size: {vocab_size}\n"
        training_info += f"Embedding dimension: {embedding_dim}\n"
        training_info += f"Context size: {context_size}\n"
        training_info += f"Final loss: {total_loss/num_batches:.4f}\n"
        training_info += f"Sample words: {', '.join(vocab[:5])}"
        
        return (center_embeddings.detach(), context_embeddings.detach(), training_info)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "NegativeSamplingNode": NegativeSamplingNode,
    "HierarchicalSoftmaxNode": HierarchicalSoftmaxNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NegativeSamplingNode": "Negative Sampling 🐱",
    "HierarchicalSoftmaxNode": "Hierarchical Softmax 🐱",
}