# RNN Module Documentation

Based on implementations from [d2l-zh (Dive into Deep Learning)](https://zh.d2l.ai/)

## Overview

The RNN module provides implementations of Recurrent Neural Networks, including basic RNNs, GRUs, and LSTMs. These nodes are based on the excellent educational resource from the d2l-zh project.

## Nodes

### 1. RNN Layer 🐱

Creates a basic Recurrent Neural Network layer.

**Parameters:**
- `input_size`: Size of input features
- `hidden_size`: Number of features in the hidden state
- `num_layers`: Number of recurrent layers
- `nonlinearity`: Activation function ('tanh' or 'relu')
- `dropout`: Dropout probability (optional)
- `bias`: Whether to use bias weights (optional)

**Outputs:**
- `rnn_model`: The RNN layer
- `rnn_info`: Information about the created layer

### 2. GRU Layer 🐱

Creates a Gated Recurrent Unit layer.

**Parameters:**
- `input_size`: Size of input features
- `hidden_size`: Number of features in the hidden state
- `num_layers`: Number of recurrent layers
- `dropout`: Dropout probability (optional)
- `bias`: Whether to use bias weights (optional)

**Outputs:**
- `gru_model`: The GRU layer
- `gru_info`: Information about the created layer

### 3. LSTM Layer 🐱

Creates a Long Short-Term Memory layer.

**Parameters:**
- `input_size`: Size of input features
- `hidden_size`: Number of features in the hidden state
- `num_layers`: Number of recurrent layers
- `dropout`: Dropout probability (optional)
- `bias`: Whether to use bias weights (optional)

**Outputs:**
- `lstm_model`: The LSTM layer
- `lstm_info`: Information about the created layer

### 4. RNN Model 🐱

Creates a complete RNN model with a linear output layer for sequence modeling.

**Parameters:**
- `rnn_layer`: The RNN layer (RNN, GRU, or LSTM)
- `vocab_size`: Size of the vocabulary for the output layer

**Outputs:**
- `rnn_model`: The complete RNN model
- `model_info`: Information about the created model

### 5. RNN Forward Pass 🐱

Performs a forward pass through an RNN model.

**Parameters:**
- `rnn_model`: The RNN model
- `input_sequence`: Input tensor sequence
- `device`: Device to run on ('cpu' or 'cuda')

**Outputs:**
- `output`: Output tensor from the model
- `final_state`: Final hidden state of the model
- `forward_info`: Information about the forward pass

### 6. RNN Test Data 🐱

Generates test data for RNN nodes.

**Parameters:**
- `sequence_length`: Length of the sequence
- `batch_size`: Size of the batch
- `input_size`: Size of input features
- `data_type`: Type of data to generate ('random' or 'indices')

**Outputs:**
- `test_data`: Generated test data tensor
- `data_info`: Information about the generated data

## Usage Examples

### Basic RNN Usage
1. Create an RNN layer using the "RNN Layer 🐱" node
2. Create a complete model using the "RNN Model 🐱" node
3. Process sequences using the "RNN Forward Pass 🐱" node

### GRU Usage
1. Create a GRU layer using the "GRU Layer 🐱" node
2. Create a complete model using the "RNN Model 🐱" node
3. Process sequences using the "RNN Forward Pass 🐱" node

### LSTM Usage
1. Create an LSTM layer using the "LSTM Layer 🐱" node
2. Create a complete model using the "RNN Model 🐱" node
3. Process sequences using the "RNN Forward Pass 🐱" node

## Example Workflow

See [example_workflow.json](../RNNs/example_workflow.json) for a complete example showing how to use all RNN nodes together, including visualization of the outputs.

## Implementation Details

These nodes are based on the PyTorch implementations described in the d2l-zh book:
- Basic RNN concepts: [Chapter 9.1](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html)
- GRU: [Chapter 10.1](https://zh.d2l.ai/chapter_recurrent-modern/gru.html)
- LSTM: [Chapter 10.2](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)

The implementations follow the time-major format (sequence_length, batch_size, feature_size) as used in the d2l examples.

The RNN Model node can handle both direct tensor inputs and index inputs (for use with one-hot encoding). When integer tensors are provided, they are automatically converted to one-hot vectors, which is useful for language modeling tasks. When float tensors are provided, they are used directly, which is useful for general sequence modeling tasks.