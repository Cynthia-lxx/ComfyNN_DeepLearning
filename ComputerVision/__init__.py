# ComputerVision module for ComfyNN_DeepLearning

# Import all node modules
from . import image_augmentation
from . import finetuning
from . import bounding_boxes
from . import anchor_boxes
from . import iou
from . import single_shot_multibox
from . import rcnn_series
from . import semantic_segmentation
from . import transposed_convolution
from . import fully_convolutional_network
from . import style_transfer

# Collect all NODE_CLASS_MAPPINGS
try:
    from .image_augmentation import NODE_CLASS_MAPPINGS as IMAGE_AUGMENTATION_MAPPINGS
except ImportError as e:
    print(f"Error importing image_augmentation mappings: {e}")
    IMAGE_AUGMENTATION_MAPPINGS = {}

try:
    from .finetuning import NODE_CLASS_MAPPINGS as FINETUNING_MAPPINGS
except ImportError as e:
    print(f"Error importing finetuning mappings: {e}")
    FINETUNING_MAPPINGS = {}

try:
    from .bounding_boxes import NODE_CLASS_MAPPINGS as BOUNDING_BOXES_MAPPINGS
except ImportError as e:
    print(f"Error importing bounding_boxes mappings: {e}")
    BOUNDING_BOXES_MAPPINGS = {}

try:
    from .anchor_boxes import NODE_CLASS_MAPPINGS as ANCHOR_BOXES_MAPPINGS
except ImportError as e:
    print(f"Error importing anchor_boxes mappings: {e}")
    ANCHOR_BOXES_MAPPINGS = {}

try:
    from .iou import NODE_CLASS_MAPPINGS as IOU_MAPPINGS
except ImportError as e:
    print(f"Error importing iou mappings: {e}")
    IOU_MAPPINGS = {}

try:
    from .single_shot_multibox import NODE_CLASS_MAPPINGS as SSD_MAPPINGS
except ImportError as e:
    print(f"Error importing single_shot_multibox mappings: {e}")
    SSD_MAPPINGS = {}

try:
    from .rcnn_series import NODE_CLASS_MAPPINGS as RCNN_MAPPINGS
except ImportError as e:
    print(f"Error importing rcnn_series mappings: {e}")
    RCNN_MAPPINGS = {}

try:
    from .semantic_segmentation import NODE_CLASS_MAPPINGS as SEGMENTATION_MAPPINGS
except ImportError as e:
    print(f"Error importing semantic_segmentation mappings: {e}")
    SEGMENTATION_MAPPINGS = {}

try:
    from .transposed_convolution import NODE_CLASS_MAPPINGS as TRANSPOSED_CONV_MAPPINGS
except ImportError as e:
    print(f"Error importing transposed_convolution mappings: {e}")
    TRANSPOSED_CONV_MAPPINGS = {}

try:
    from .fully_convolutional_network import NODE_CLASS_MAPPINGS as FCN_MAPPINGS
except ImportError as e:
    print(f"Error importing fully_convolutional_network mappings: {e}")
    FCN_MAPPINGS = {}

try:
    from .style_transfer import NODE_CLASS_MAPPINGS as STYLE_TRANSFER_MAPPINGS
except ImportError as e:
    print(f"Error importing style_transfer mappings: {e}")
    STYLE_TRANSFER_MAPPINGS = {}

# Merge all mappings
NODE_CLASS_MAPPINGS = {
    **IMAGE_AUGMENTATION_MAPPINGS,
    **FINETUNING_MAPPINGS,
    **BOUNDING_BOXES_MAPPINGS,
    **ANCHOR_BOXES_MAPPINGS,
    **IOU_MAPPINGS,
    **SSD_MAPPINGS,
    **RCNN_MAPPINGS,
    **SEGMENTATION_MAPPINGS,
    **TRANSPOSED_CONV_MAPPINGS,
    **FCN_MAPPINGS,
    **STYLE_TRANSFER_MAPPINGS
}

# Collect all NODE_DISPLAY_NAME_MAPPINGS
try:
    from .image_augmentation import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_AUGMENTATION_NAMES
except ImportError as e:
    print(f"Error importing image_augmentation display names: {e}")
    IMAGE_AUGMENTATION_NAMES = {}

try:
    from .finetuning import NODE_DISPLAY_NAME_MAPPINGS as FINETUNING_NAMES
except ImportError as e:
    print(f"Error importing finetuning display names: {e}")
    FINETUNING_NAMES = {}

try:
    from .bounding_boxes import NODE_DISPLAY_NAME_MAPPINGS as BOUNDING_BOXES_NAMES
except ImportError as e:
    print(f"Error importing bounding_boxes display names: {e}")
    BOUNDING_BOXES_NAMES = {}

try:
    from .anchor_boxes import NODE_DISPLAY_NAME_MAPPINGS as ANCHOR_BOXES_NAMES
except ImportError as e:
    print(f"Error importing anchor_boxes display names: {e}")
    ANCHOR_BOXES_NAMES = {}

try:
    from .iou import NODE_DISPLAY_NAME_MAPPINGS as IOU_NAMES
except ImportError as e:
    print(f"Error importing iou display names: {e}")
    IOU_NAMES = {}

try:
    from .single_shot_multibox import NODE_DISPLAY_NAME_MAPPINGS as SSD_NAMES
except ImportError as e:
    print(f"Error importing single_shot_multibox display names: {e}")
    SSD_NAMES = {}

try:
    from .rcnn_series import NODE_DISPLAY_NAME_MAPPINGS as RCNN_NAMES
except ImportError as e:
    print(f"Error importing rcnn_series display names: {e}")
    RCNN_NAMES = {}

try:
    from .semantic_segmentation import NODE_DISPLAY_NAME_MAPPINGS as SEGMENTATION_NAMES
except ImportError as e:
    print(f"Error importing semantic_segmentation display names: {e}")
    SEGMENTATION_NAMES = {}

try:
    from .transposed_convolution import NODE_DISPLAY_NAME_MAPPINGS as TRANSPOSED_CONV_NAMES
except ImportError as e:
    print(f"Error importing transposed_convolution display names: {e}")
    TRANSPOSED_CONV_NAMES = {}

try:
    from .fully_convolutional_network import NODE_DISPLAY_NAME_MAPPINGS as FCN_NAMES
except ImportError as e:
    print(f"Error importing fully_convolutional_network display names: {e}")
    FCN_NAMES = {}

try:
    from .style_transfer import NODE_DISPLAY_NAME_MAPPINGS as STYLE_TRANSFER_NAMES
except ImportError as e:
    print(f"Error importing style_transfer display names: {e}")
    STYLE_TRANSFER_NAMES = {}

# Merge all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    **IMAGE_AUGMENTATION_NAMES,
    **FINETUNING_NAMES,
    **BOUNDING_BOXES_NAMES,
    **ANCHOR_BOXES_NAMES,
    **IOU_NAMES,
    **SSD_NAMES,
    **RCNN_NAMES,
    **SEGMENTATION_NAMES,
    **TRANSPOSED_CONV_NAMES,
    **FCN_NAMES,
    **STYLE_TRANSFER_NAMES
}