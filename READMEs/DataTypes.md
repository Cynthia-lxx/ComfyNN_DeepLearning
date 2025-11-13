# DataTypes Module

The DataTypes module provides functionality for converting between different data types and creating tensors.

## Features

### Creation Nodes
- TensorCreator: Create tensors with specified shape and values
- RandomTensorCreator: Create tensors with random values
- SpecialTensorCreator: Create special tensors (zeros, ones, identity matrices)
- TensorReshaper: Reshape existing tensors
- TensorConverter: Convert between different tensor types

### DataTypes Nodes
- ImageToTensor: Convert images to tensors
- TensorToImage: Convert tensors to images
- ModelToTensor: Extract model parameters as tensors
- TensorToModel: Apply tensors as model parameters
- ClipToTensor: Convert CLIP embeddings to tensors
- TensorToClip: Convert tensors to CLIP embeddings
- VaeToTensor: Extract VAE parameters as tensors
- TensorToVae: Apply tensors as VAE parameters
- LoadTensor: Load tensors from files
- SaveTensor: Save tensors to files

## Example Workflow

See `example_workflow.json` in the DataTypes directory for a demonstration of how to use these nodes.