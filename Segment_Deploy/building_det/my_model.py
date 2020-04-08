import time
now = time.time()

import os
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.draw
import json

# Root directory of the project
ROOT_DIR = os.path.abspath("")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config, AerialConfig, InferenceConfig
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.utils import AerialDataset

# check config parameters from iteration file
config = AerialConfig()
inference_config = InferenceConfig()

# improve model and append path here
COCO_MODEL_PATH_TRAINED = 'D:\\Hamzah\\JobPrep\\GPU_Projects\\MapDataset\\MatterPortGit\\Mask_RCNN\\Our_Implementation\\mask_rcnn_aerial_0030.h5'

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",config=inference_config, model_dir=COCO_MODEL_PATH_TRAINED)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()

model_path = COCO_MODEL_PATH_TRAINED

# Load trained weights
print("Loading weights from ", model_path)

model.load_weights(COCO_MODEL_PATH_TRAINED, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

class_names = ['BG', 'building', 'pool']

image_file = "D:\\Hamzah\\JobPrep\\GPU_Projects\\MapDataset\\MatterPortGit\\Mask_RCNN\\Our_Implementation\\aerial\\000000033121.jpg"
image = skimage.io.imread(image_file)
r = model.detect([image], verbose=1)[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

print(time.time() - now)

