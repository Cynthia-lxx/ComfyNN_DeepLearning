# ComfyNN NLP Pretrain Word Embeddings Nodes
# Based on d2l-zh implementation (https://github.com/d2l-ai/d2l-zh)
# Thank you d2l-ai team for the excellent educational resource

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import Counter
import math

class Word2VecSelfSupervised:
    """Word2Vec self-supervised learning node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_corpus": ("STRING", {"multiline": True, "default": "the man loves his son\nthe cat sat on the mat"}),
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 5}),
            },
            "optional": {
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "num_epochs": ("INT", {"default": 100, "min": 10, "max": 1000}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("center_embeddings", "context_embeddings", "training_info")
    FUNCTION = "train_word2vec"
    CATEGORY = "ComfyNN/NLP_Pretrain/WordEmbeddings"
    DESCRIPTION = "Train Word2Vec embeddings using self-supervised learning"

    def train_word2vec(self, text_corpus, embedding_dim, window_size, learning_rate=0.01, num_epochs=100):
        # Preprocess text
        sentences = text_corpus.strip().split('\n')
        tokenized_sentences = [sentence.split() for sentence in sentences]
        
        # Build vocabulary
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        vocab = list(set(all_words))
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_word = {i: word for i, word in enumerate(vocab)}
        
        # Generate training data
        center_words = []
        context_words = []
        
        for sentence in tokenized_sentences:
            for i, center_word in enumerate(sentence):
                center_idx = word_to_idx[center_word]
                # Get context words within window
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
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
        context_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        
        # Training
        optimizer = torch.optim.SGD([center_embeddings, context_embeddings], lr=learning_rate)
        
        for epoch in range(num_epochs):
            # Forward pass
            center_embeds = center_embeddings[center_words]  # [N, embedding_dim]
            context_embeds = context_embeddings[context_words]  # [N, embedding_dim]
            
            # Compute scores
            scores = torch.sum(center_embeds * context_embeds, dim=1)  # [N]
            
            # Compute loss (negative sampling would be better, but using full softmax for simplicity)
            # For numerical stability, we compute log P(context|center) = score - logsumexp(scores_all)
            all_scores = torch.matmul(center_embeds, context_embeddings.t())  # [N, vocab_size]
            log_denominator = torch.logsumexp(all_scores, dim=1)  # [N]
            log_prob = scores - log_denominator  # [N]
            loss = -torch.mean(log_prob)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Generate training info
        training_info = f"Word2Vec Training Completed\n"
        training_info += f"Vocabulary size: {vocab_size}\n"
        training_info += f"Embedding dimension: {embedding_dim}\n"
        training_info += f"Training samples: {len(center_words)}\n"
        training_info += f"Final loss: {loss.item():.4f}\n"
        training_info += f"Sample words: {', '.join(vocab[:5])}"
        
        return (center_embeddings.detach(), context_embeddings.detach(), training_info)


class SkipGramModel:
    """Skip-Gram Model node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_corpus": ("STRING", {"multiline": True, "default": "the man loves his son\nthe cat sat on the mat"}),
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 5}),
            },
            "optional": {
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "num_epochs": ("INT", {"default": 100, "min": 10, "max": 1000}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("center_embeddings", "context_embeddings", "training_info")
    FUNCTION = "train_skipgram"
    CATEGORY = "ComfyNN/NLP_Pretrain/WordEmbeddings"
    DESCRIPTION = "Train Skip-Gram model for word embeddings"

    def train_skipgram(self, text_corpus, embedding_dim, window_size, learning_rate=0.01, num_epochs=100):
        # This is essentially the same as Word2Vec implementation above
        # Skip-Gram assumes a center word generates context words
        return Word2VecSelfSupervised().train_word2vec(
            text_corpus, embedding_dim, window_size, learning_rate, num_epochs
        )


class CBOWModel:
    """Continuous Bag of Words Model node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_corpus": ("STRING", {"multiline": True, "default": "the man loves his son\nthe cat sat on the mat"}),
                "embedding_dim": ("INT", {"default": 100, "min": 10, "max": 500}),
                "window_size": ("INT", {"default": 2, "min": 1, "max": 5}),
            },
            "optional": {
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "num_epochs": ("INT", {"default": 100, "min": 10, "max": 1000}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("context_embeddings", "center_embeddings", "training_info")
    FUNCTION = "train_cbow"
    CATEGORY = "ComfyNN/NLP_Pretrain/WordEmbeddings"
    DESCRIPTION = "Train Continuous Bag of Words model for word embeddings"

    def train_cbow(self, text_corpus, embedding_dim, window_size, learning_rate=0.01, num_epochs=100):
        # Preprocess text
        sentences = text_corpus.strip().split('\n')
        tokenized_sentences = [sentence.split() for sentence in sentences]
        
        # Build vocabulary
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        vocab = list(set(all_words))
        vocab_size = len(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_word = {i: word for i, word in enumerate(vocab)}
        
        # Generate training data
        context_word_groups = []  # List of lists of context words
        target_words = []  # List of target words
        
        for sentence in tokenized_sentences:
            for i, target_word in enumerate(sentence):
                target_idx = word_to_idx[target_word]
                # Get context words within window
                context_indices = []
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                for j in range(start, end):
                    if j != i:
                        context_indices.append(word_to_idx[sentence[j]])
                
                if context_indices:  # Only add if we have context words
                    context_word_groups.append(context_indices)
                    target_words.append(target_idx)
        
        # Convert to tensors (pad context groups to same length)
        max_context_len = max(len(group) for group in context_word_groups)
        padded_contexts = []
        for group in context_word_groups:
            # Pad with -1 (will be handled in training)
            padded_group = group + [-1] * (max_context_len - len(group))
            padded_contexts.append(padded_group)
        
        context_words = torch.tensor(padded_contexts, dtype=torch.long)  # [N, max_context_len]
        target_words = torch.tensor(target_words, dtype=torch.long)  # [N]
        
        # Initialize embeddings
        context_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        center_embeddings = torch.randn(vocab_size, embedding_dim, requires_grad=True)
        
        # Training
        optimizer = torch.optim.SGD([context_embeddings, center_embeddings], lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(len(target_words)):
                target_idx = target_words[i]
                context_indices = context_words[i]
                
                # Filter out padding
                valid_context_indices = [idx for idx in context_indices if idx != -1]
                if not valid_context_indices:
                    continue
                
                # Average context embeddings
                context_embeds = context_embeddings[valid_context_indices]  # [C, embedding_dim]
                avg_context_embed = torch.mean(context_embeds, dim=0, keepdim=True)  # [1, embedding_dim]
                
                # Compute scores with all vocabulary
                all_scores = torch.matmul(avg_context_embed, center_embeddings.t())  # [1, vocab_size]
                
                # Compute loss
                log_denominator = torch.logsumexp(all_scores, dim=1)  # [1]
                target_score = all_scores[0, target_idx]  # scalar
                log_prob = target_score - log_denominator  # [1]
                loss = -log_prob
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Generate training info
        avg_loss = total_loss / len(target_words) if len(target_words) > 0 else 0
        training_info = f"CBOW Training Completed\n"
        training_info += f"Vocabulary size: {vocab_size}\n"
        training_info += f"Embedding dimension: {embedding_dim}\n"
        training_info += f"Training samples: {len(target_words)}\n"
        training_info += f"Final loss: {avg_loss:.4f}\n"
        training_info += f"Sample words: {', '.join(vocab[:5])}"
        
        return (context_embeddings.detach(), center_embeddings.detach(), training_info)


class SubsamplingNLP:
    """Subsampling for NLP node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_corpus": ("STRING", {"multiline": True, "default": "the man loves his son\nthe cat sat on the mat"}),
                "subsampling_rate": ("FLOAT", {"default": 1e-5, "min": 1e-7, "max": 1e-3, "step": 1e-6}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("subsampled_text", "subsampling_info")
    FUNCTION = "subsample"
    CATEGORY = "ComfyNN/NLP_Pretrain/WordEmbeddings"
    DESCRIPTION = "Perform subsampling on text corpus to reduce frequent word impact"

    def subsample(self, text_corpus, subsampling_rate):
        # Preprocess text
        sentences = text_corpus.strip().split('\n')
        tokenized_sentences = [sentence.split() for sentence in sentences]
        
        # Count word frequencies
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        word_counts = Counter(all_words)
        total_words = len(all_words)
        
        # Calculate subsampling probabilities
        word_probs = {}
        for word, count in word_counts.items():
            freq = count / total_words
            prob = (math.sqrt(freq / subsampling_rate) + 1) * (subsampling_rate / freq)
            word_probs[word] = min(prob, 1.0)
        
        # Apply subsampling
        subsampled_sentences = []
        kept_words = 0
        total_words_after = 0
        
        for sentence in tokenized_sentences:
            subsampled_sentence = []
            for word in sentence:
                if random.random() < word_probs.get(word, 1.0):
                    subsampled_sentence.append(word)
                    kept_words += 1
                total_words_after += 1
            
            if subsampled_sentence:  # Only add non-empty sentences
                subsampled_sentences.append(' '.join(subsampled_sentence))
        
        subsampled_text = '\n'.join(subsampled_sentences)
        
        # Generate info
        subsampling_info = f"Subsampling Completed\n"
        subsampling_info += f"Original words: {total_words}\n"
        subsampling_info += f"Words after subsampling: {kept_words}\n"
        subsampling_info += f"Subsampling rate: {subsampling_rate}\n"
        subsampling_info += f"Reduction ratio: {kept_words/total_words:.2%}\n"
        subsampling_info += f"Sample word probabilities:\n"
        for word in list(word_counts.keys())[:5]:
            subsampling_info += f"  {word}: {word_probs.get(word, 1.0):.4f}\n"
        
        return (subsampled_text, subsampling_info)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "Word2VecSelfSupervised": Word2VecSelfSupervised,
    "SkipGramModel": SkipGramModel,
    "CBOWModel": CBOWModel,
    "SubsamplingNLP": SubsamplingNLP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Word2VecSelfSupervised": "Word2Vec Self-Supervised 🐱",
    "SkipGramModel": "Skip-Gram Model 🐱",
    "CBOWModel": "CBOW Model 🐱",
    "SubsamplingNLP": "Subsampling NLP 🐱",
}