# ComfyNN RNN Nodes
# Based on d2l-zh implementation (https://github.com/d2l-ai/d2l-zh)
# Thank you d2l-ai team for the excellent educational resource

import torch
import torch.nn as nn
import numpy as np

class ComfyNNRNNNode:
    """Basic RNN node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_size": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "hidden_size": ("INT", {"default": 128, "min": 1, "max": 512}),
                "num_layers": ("INT", {"default": 1, "min": 1, "max": 10}),
                "nonlinearity": (["tanh", "relu"], {"default": "tanh"}),
            },
            "optional": {
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
                "bias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CUSTOM", "STRING")
    RETURN_NAMES = ("rnn_model", "rnn_info")
    FUNCTION = "create_rnn"
    CATEGORY = "ComfyNN/RNNs"
    DESCRIPTION = "Create a basic RNN layer based on d2l-zh implementation"

    def create_rnn(self, input_size, hidden_size, num_layers, nonlinearity, dropout=0.0, bias=True):
        try:
            rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                dropout=dropout,
                bias=bias,
                batch_first=False  # time_major=True like in d2l
            )
            
            rnn_info = f"Basic RNN Layer 🐱\n"
            rnn_info += f"Input size: {input_size}\n"
            rnn_info += f"Hidden size: {hidden_size}\n"
            rnn_info += f"Number of layers: {num_layers}\n"
            rnn_info += f"Nonlinearity: {nonlinearity}\n"
            rnn_info += f"Dropout: {dropout}\n"
            rnn_info += f"Bias: {bias}"
            
            return (rnn, rnn_info)
        except Exception as e:
            return (None, f"Error creating RNN: {str(e)}")


class ComfyNNGRUNode:
    """GRU node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_size": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "hidden_size": ("INT", {"default": 128, "min": 1, "max": 512}),
                "num_layers": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
                "bias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CUSTOM", "STRING")
    RETURN_NAMES = ("gru_model", "gru_info")
    FUNCTION = "create_gru"
    CATEGORY = "ComfyNN/RNNs"
    DESCRIPTION = "Create a GRU layer based on d2l-zh implementation"

    def create_gru(self, input_size, hidden_size, num_layers, dropout=0.0, bias=True):
        try:
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bias=bias,
                batch_first=False  # time_major=True like in d2l
            )
            
            gru_info = f"GRU Layer 🐱\n"
            gru_info += f"Input size: {input_size}\n"
            gru_info += f"Hidden size: {hidden_size}\n"
            gru_info += f"Number of layers: {num_layers}\n"
            gru_info += f"Dropout: {dropout}\n"
            gru_info += f"Bias: {bias}"
            
            return (gru, gru_info)
        except Exception as e:
            return (None, f"Error creating GRU: {str(e)}")


class ComfyNNLSTMNode:
    """LSTM node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_size": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "hidden_size": ("INT", {"default": 128, "min": 1, "max": 512}),
                "num_layers": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {
                "dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05}),
                "bias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CUSTOM", "STRING")
    RETURN_NAMES = ("lstm_model", "lstm_info")
    FUNCTION = "create_lstm"
    CATEGORY = "ComfyNN/RNNs"
    DESCRIPTION = "Create an LSTM layer based on d2l-zh implementation"

    def create_lstm(self, input_size, hidden_size, num_layers, dropout=0.0, bias=True):
        try:
            lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bias=bias,
                batch_first=False  # time_major=True like in d2l
            )
            
            lstm_info = f"LSTM Layer 🐱\n"
            lstm_info += f"Input size: {input_size}\n"
            lstm_info += f"Hidden size: {hidden_size}\n"
            lstm_info += f"Number of layers: {num_layers}\n"
            lstm_info += f"Dropout: {dropout}\n"
            lstm_info += f"Bias: {bias}"
            
            return (lstm, lstm_info)
        except Exception as e:
            return (None, f"Error creating LSTM: {str(e)}")


class ComfyNNRNNModelNode:
    """RNN Model node for sequence modeling based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rnn_layer": ("CUSTOM",),
                "vocab_size": ("INT", {"default": 1000, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("CUSTOM", "STRING")
    RETURN_NAMES = ("rnn_model", "model_info")
    FUNCTION = "create_model"
    CATEGORY = "ComfyNN/RNNs"
    DESCRIPTION = "Create a complete RNN model with linear output layer based on d2l-zh implementation"

    def create_model(self, rnn_layer, vocab_size):
        try:
            # Determine hidden size from the RNN layer
            hidden_size = rnn_layer.hidden_size
            num_directions = 2 if rnn_layer.bidirectional else 1
            
            # Create the model class
            class RNNModel(nn.Module):
                def __init__(self, rnn_layer, vocab_size):
                    super(RNNModel, self).__init__()
                    self.rnn = rnn_layer
                    self.vocab_size = vocab_size
                    self.num_hiddens = self.rnn.hidden_size
                    # If RNN is bidirectional, num_directions should be 2, else 1
                    if not self.rnn.bidirectional:
                        self.num_directions = 1
                        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
                    else:
                        self.num_directions = 2
                        self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

                def forward(self, inputs, state):
                    # For direct tensor inputs (not indices), we don't need one-hot encoding
                    if inputs.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
                        # One-hot encoding of inputs if they are indices
                        X = nn.functional.one_hot(inputs.long().T, self.vocab_size)
                        X = X.to(torch.float32)
                    else:
                        # Direct use of input tensors
                        X = inputs
                    Y, state = self.rnn(X, state)
                    # The fully connected layer first changes Y to the shape
                    # (time_steps * batch_size, hidden_units)
                    # Its output shape is (time_steps * batch_size, vocab_size)
                    output = self.linear(Y.reshape((-1, Y.shape[-1])))
                    return output, state

                def begin_state(self, device, batch_size=1):
                    if not isinstance(self.rnn, nn.LSTM):
                        # nn.GRU takes a tensor as hidden state
                        return torch.zeros((self.num_directions * self.rnn.num_layers,
                                           batch_size, self.num_hiddens), 
                                          device=device)
                    else:
                        # nn.LSTM takes a tuple as hidden state
                        return (torch.zeros((
                            self.num_directions * self.rnn.num_layers,
                            batch_size, self.num_hiddens), device=device),
                                torch.zeros((
                                    self.num_directions * self.rnn.num_layers,
                                    batch_size, self.num_hiddens), device=device))
            
            # Create the model instance
            model = RNNModel(rnn_layer, vocab_size)
            
            model_info = f"RNN Model 🐱\n"
            model_info += f"Vocabulary size: {vocab_size}\n"
            model_info += f"Hidden size: {hidden_size}\n"
            model_info += f"Number of directions: {num_directions}"
            
            return (model, model_info)
        except Exception as e:
            return (None, f"Error creating RNN model: {str(e)}")


class ComfyNNRNNForwardNode:
    """RNN Forward Pass node for processing sequences"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rnn_model": ("CUSTOM",),
                "input_sequence": ("TENSOR",),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("TENSOR", "CUSTOM", "STRING")
    RETURN_NAMES = ("output", "final_state", "forward_info")
    FUNCTION = "forward_pass"
    CATEGORY = "ComfyNN/RNNs"
    DESCRIPTION = "Perform forward pass through RNN model"

    def forward_pass(self, rnn_model, input_sequence, device):
        try:
            # Move model and input to device
            rnn_model = rnn_model.to(device)
            input_sequence = input_sequence.to(device)
            
            # Initialize state
            batch_size = input_sequence.shape[1] if len(input_sequence.shape) > 1 else 1
            state = rnn_model.begin_state(device, batch_size)
            
            # Forward pass
            output, final_state = rnn_model(input_sequence, state)
            
            forward_info = f"RNN Forward Pass 🐱\n"
            forward_info += f"Input shape: {list(input_sequence.shape)}\n"
            forward_info += f"Output shape: {list(output.shape)}"
            
            # Ensure output is not None
            if output is None:
                return (torch.zeros(1), final_state, f"Error in forward pass: output is None\n{forward_info}")
            
            return (output, final_state, forward_info)
        except Exception as e:
            return (torch.zeros(1), None, f"Error in forward pass: {str(e)}")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ComfyNNRNNNode": ComfyNNRNNNode,
    "ComfyNNGRUNode": ComfyNNGRUNode,
    "ComfyNNLSTMNode": ComfyNNLSTMNode,
    "ComfyNNRNNModelNode": ComfyNNRNNModelNode,
    "ComfyNNRNNForwardNode": ComfyNNRNNForwardNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyNNRNNNode": "RNN Layer 🐱",
    "ComfyNNGRUNode": "GRU Layer 🐱",
    "ComfyNNLSTMNode": "LSTM Layer 🐱",
    "ComfyNNRNNModelNode": "RNN Model 🐱",
    "ComfyNNRNNForwardNode": "RNN Forward Pass 🐱",
}