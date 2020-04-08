from . import mrcnn
from .mrcnn.config import Config, AerialConfig, InferenceConfig
from .mrcnn import utils
from .mrcnn import model as modellib
from .mrcnn import visualize
from .mrcnn.model import log
from .mrcnn.utils import AerialDataset

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

# check config parameters from iteration file
inference_config = InferenceConfig()
# improve model and append path here

COCO_MODEL_PATH_TRAINED = 'D:\\Hamzah\\JobPrep\\GPU_Projects\\MapDataset\\MatterPortGit\\Mask_RCNN\\Our_Implementation\\mask_rcnn_aerial_0030.h5'
# Recreate the model in inference mode

model = modellib.MaskRCNN(mode="inference",config=inference_config, model_dir=COCO_MODEL_PATH_TRAINED)
model.load_weights(COCO_MODEL_PATH_TRAINED, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

CLASS_NAMES = ['BG', 'building', 'pool']

print("Iported model")
