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
from .image_augmentation import NODE_CLASS_MAPPINGS as IMAGE_AUGMENTATION_MAPPINGS
from .finetuning import NODE_CLASS_MAPPINGS as FINETUNING_MAPPINGS
from .bounding_boxes import NODE_CLASS_MAPPINGS as BOUNDING_BOXES_MAPPINGS
from .anchor_boxes import NODE_CLASS_MAPPINGS as ANCHOR_BOXES_MAPPINGS
from .iou import NODE_CLASS_MAPPINGS as IOU_MAPPINGS
from .single_shot_multibox import NODE_CLASS_MAPPINGS as SSD_MAPPINGS
from .rcnn_series import NODE_CLASS_MAPPINGS as RCNN_MAPPINGS
from .semantic_segmentation import NODE_CLASS_MAPPINGS as SEGMENTATION_MAPPINGS
from .transposed_convolution import NODE_CLASS_MAPPINGS as TRANSPOSED_CONV_MAPPINGS
from .fully_convolutional_network import NODE_CLASS_MAPPINGS as FCN_MAPPINGS
from .style_transfer import NODE_CLASS_MAPPINGS as STYLE_TRANSFER_MAPPINGS

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
from .image_augmentation import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_AUGMENTATION_NAMES
from .finetuning import NODE_DISPLAY_NAME_MAPPINGS as FINETUNING_NAMES
from .bounding_boxes import NODE_DISPLAY_NAME_MAPPINGS as BOUNDING_BOXES_NAMES
from .anchor_boxes import NODE_DISPLAY_NAME_MAPPINGS as ANCHOR_BOXES_NAMES
from .iou import NODE_DISPLAY_NAME_MAPPINGS as IOU_NAMES
from .single_shot_multibox import NODE_DISPLAY_NAME_MAPPINGS as SSD_NAMES
from .rcnn_series import NODE_DISPLAY_NAME_MAPPINGS as RCNN_NAMES
from .semantic_segmentation import NODE_DISPLAY_NAME_MAPPINGS as SEGMENTATION_NAMES
from .transposed_convolution import NODE_DISPLAY_NAME_MAPPINGS as TRANSPOSED_CONV_NAMES
from .fully_convolutional_network import NODE_DISPLAY_NAME_MAPPINGS as FCN_NAMES
from .style_transfer import NODE_DISPLAY_NAME_MAPPINGS as STYLE_TRANSFER_NAMES

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
