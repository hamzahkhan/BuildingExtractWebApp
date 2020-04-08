# BuildingExtractWebApp
A web app in Django with trained MRCNN architecture on the backend

Django project named Segment_deploy. I tried to build a web app on top of my MRCNN tuned architecture to extract building images from VHR satellite images.
Error is famous one actually :D  <br />
Error link - https://github.com/tensorflow/tensorflow/issues/14356 <br />
Possible solution - https://github.com/tensorflow/tensorflow/issues/14356#issuecomment-385962623 <br />
The solution is to load the tensorflow default graph and load a RESNET50 model with it. It fails the whole process of my trained model. So no success in deploying. ;)
The possible solution works for some images of (x,y,4) format.
