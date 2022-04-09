from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

classfile = "coco.names"
# imagepath = 'test/6.jpg'
videopath = 'test/9.mp4'
# videopath = 0 # for webcam
threshold = 0.5

detector = Detector()
detector.readClasses(classfile)
detector.downloadModel(modelURL)
detector.loadModel()
# detector.predictImage(imagepath, threshold)
detector.predictVideo(videopath, threshold)