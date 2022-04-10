import os
from Detector import *

# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_resnet50_v1_fpn_512x512_coco17_tpu-8.tar.gz"

classfile = "coco.names"
# imagepath = 'test/7.jpg'
videopath = 'test/9.mp4'
# videopath = 0 # for webcam
threshold = 0.5

detector = Detector()
detector.readClasses(classfile)
detector.downloadModel(modelURL)
detector.loadModel()
# detector.predictImage(imagepath, threshold)
detector.predictVideo(videopath, threshold)

# # get all file names in test folder
# filenames = next(os.walk("test"), (None, None, []))[2]
# for f in filenames:
#     extension = f[-4:]
#     if extension == ".jpg":
#         detector.predictImage(os.path.join("test", f), threshold)
#     elif extension == ".mp4":
#         detector.predictVideo(os.path.join("test", f), threshold)