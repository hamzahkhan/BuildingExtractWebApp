image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,    minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
data.update({"num_faces": len(rects), "faces": rects, "success": True})

############################
image_file = "D:\\Hamzah\\JobPrep\\GPU_Projects\\MapDataset\\MatterPortGit\\Mask_RCNN\\Our_Implementation\\aerial\\000000033121.jpg"
image = skimage.io.imread(image_file)

config = AerialConfig()
inference_config = InferenceConfig()
COCO_MODEL_PATH_TRAINED = 'D:\\Hamzah\\JobPrep\\GPU_Projects\\MapDataset\\MatterPortGit\\Mask_RCNN\\Our_Implementation\\mask_rcnn_aerial_0030.h5'
model = modellib.MaskRCNN(mode="inference",config=inference_config, model_dir=COCO_MODEL_PATH_TRAINED)
model.load_weights(COCO_MODEL_PATH_TRAINED, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

class_names = ['BG', 'building', 'pool']
r = model.detect([image], verbose=1)[0]
data.update({'rois': r['rois'],'masks': r['masks'],'class_ids': r['class_ids'], 'class_names':class_names,'scores': r['scores']})

#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

#model_path = COCO_MODEL_PATH_TRAINED

# Load trained weights
#print("Loading weights from ", model_path)

