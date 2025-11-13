# DLCompute Module

The DLCompute module provides nodes for deep learning computations, including data processing, model layers, loss functions, and optimizers.

## Features

### Data Processing
- TensorDataLoader: Load and batch data for training
- TensorNormalize: Normalize tensor values
- TensorAugment: Apply data augmentation techniques

### Model Layers
- TensorLinearLayer: Apply linear transformation
- TensorConv2DLayer: Apply 2D convolution operation
- TensorActivation: Apply activation functions

### Loss Functions
- TensorMSELoss: Compute Mean Squared Error loss
- TensorCrossEntropyLoss: Compute Cross Entropy loss

### Optimizers
- TensorSGDOptimizer: Stochastic Gradient Descent optimizer
- TensorAdamOptimizer: Adam optimizer

### Training Utilities
- TensorForwardPass: Perform forward pass through a model
- TensorBackwardPass: Perform backward pass to compute gradients
- TensorUpdateWeights: Update model weights using computed gradients

### Test Data Generation
- DLComputeTestDataGenerator: Generate sample data for testing workflows

## Example Workflow

See `example_workflow.json` in the DLCompute directory for a demonstration of how to use these nodes to build a complete training pipeline.