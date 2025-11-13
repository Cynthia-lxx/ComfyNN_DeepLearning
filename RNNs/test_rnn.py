# Test nodes for RNN module
import torch
import torch.nn as nn

class ComfyNNRNNTestNode:
    """Test node for RNN module"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sequence_length": ("INT", {"default": 10, "min": 1, "max": 100}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 128}),
                "input_size": ("INT", {"default": 64, "min": 1, "max": 512}),
                "data_type": (["random", "indices"], {"default": "random"}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("test_data", "data_info")
    FUNCTION = "generate_test_data"
    CATEGORY = "ComfyNN/RNNs/Test"
    DESCRIPTION = "Generate test data for RNN nodes"

    def generate_test_data(self, sequence_length, batch_size, input_size, data_type):
        if data_type == "indices":
            # Generate integer indices for use with one-hot encoding
            test_data = torch.randint(0, input_size, (sequence_length, batch_size))
        else:
            # Generate random test data
            test_data = torch.randn(sequence_length, batch_size, input_size)
        
        data_info = f"Test Data 🐱\n"
        data_info += f"Shape: {list(test_data.shape)}\n"
        data_info += f"Sequence length: {sequence_length}\n"
        data_info += f"Batch size: {batch_size}\n"
        data_info += f"Input size: {input_size}\n"
        data_info += f"Data type: {data_type}"
        
        return (test_data, data_info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ComfyNNRNNTestNode": ComfyNNRNNTestNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyNNRNNTestNode": "RNN Test Data 🐱",
}