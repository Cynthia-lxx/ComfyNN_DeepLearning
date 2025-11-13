from .rnn_nodes import (
    ComfyNNRNNNode,
    ComfyNNGRUNode,
    ComfyNNLSTMNode,
    ComfyNNRNNModelNode,
    ComfyNNRNNForwardNode,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS
)

from .test_rnn import (
    ComfyNNRNNTestNode,
)

# Update the node mappings to include test nodes
NODE_CLASS_MAPPINGS.update({
    "ComfyNNRNNTestNode": ComfyNNRNNTestNode,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "ComfyNNRNNTestNode": "RNN Test Data 🐱",
})

__all__ = [
    "ComfyNNRNNNode",
    "ComfyNNGRUNode",
    "ComfyNNLSTMNode",
    "ComfyNNRNNModelNode",
    "ComfyNNRNNForwardNode",
    "ComfyNNRNNTestNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]