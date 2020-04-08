from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

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
import urllib

from . import model, CLASS_NAMES, ResNet50, tf
from .forms import UploadFileForm


@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}
    # check to see if this is a post request
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        print(form.is_multipart())
        print("Before post methid")
        if form.is_valid():
            # form.cleaned_data[]
            print("POST method confirmed")
            # check to see if an image was uploaded
            if request.FILES.get("image", None) is not None:
                print("Got image path")
                # grab the uploaded image
                #image = _grab_image(form.cleaned_data["url"])
                image = _grab_image(path=request.FILES["image"])
            # otherwise, assume that a URL was passed in
            else:
                # grab the URL from the request
                url = request.POST.get("url", None)
                print(url)
                # if the URL is None, then return an error
                if url is None:
                    data["error"] = "No URL provided."
                    return JsonResponse(data)
                # load the image and convert
                image = _grab_image(url=url)
               
            #model.keras_model._make_predict_function()
			
            #### ERROR while deploying to webAPP Tensor Tensor("mrcnn_detection/Reshape_1:0", shape=(1, 100, 6), dtype=float32) is not an element of this graph.

            def load_model():
                #global model
                #model = ResNet50(weights="imagenet")
            # this is key : save the graph after loading the model
                global graph
                graph = tf.get_default_graph()

            load_model()

            with graph.as_default():
                #preds = model.predict(image)
                r = model.detect([image], verbose=1)[0]
            #r = model.detect([image], verbose=1)[0]
            data.update({'rois': r['rois'], 'masks': r['masks'], 'class_ids': r['class_ids'],
                         'class_names': CLASS_NAMES, 'scores': r['scores']})
            data["success"] = True
            form = UploadFileForm()
            return render(request, 'upload.html', {'form': form, 'data': data})
        # return a JSON response
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})


def _grab_image(path=None, stream=None, url=None):
    if path is not None:
        #image = skimage.imread(path)
        image = skimage.io.imread(path)
    else:
        if url is not None:
            image = skimage.io.imread(url)
    return image
