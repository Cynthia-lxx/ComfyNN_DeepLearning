# NLP_Pretrain Module

The NLP_Pretrain module provides nodes for natural language processing pre-training tasks, including various word embedding techniques and language models.

## Features

### Word Embeddings
- Word2VecSelfSupervised: Self-supervised Word2Vec implementation
- SkipGramModel: Skip-gram model for word embeddings
- CBOWModel: Continuous Bag of Words model for word embeddings
- SubsamplingNLP: Subsampling techniques for NLP data

### Approximate Training Methods
- NegativeSamplingNLP: Negative sampling technique
- HierarchicalSoftmaxNLP: Hierarchical softmax implementation

### Advanced Embeddings
- GloVeModel: Global Vectors for Word Representation
- FastTextModel: FastText word representation model

### Transformer Models
- BERTModel: Bidirectional Encoder Representations from Transformers
- BERTMaskedLanguageModel: BERT model for masked language modeling

### Test Data Generation
- NLPTestDataGenerator: Generate sample text data for testing workflows

## Example Workflow

See `example_workflow.json` in the NLP_Pretrain directory for a demonstration of how to use these nodes.