import time
import os

from tensorflow.python.keras.utils.data_utils import get_file
import cv2
import numpy as np
import tensorflow as tf

np.random.seed(1)

class Detector:
    def __init__(self) -> None:
        pass
    
    def readClasses(self, classesFilePath):
        """
        Read the class names from classesFilePath, save the class names in a classesList.
        """
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        # get color for each class names
        # size = (num of classes available, 3 = for 3 color channels)
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

        print(len(self.classesList), len(self.colorList))
    
    def downloadModel(self, modelURL):
        """
        Download Tensorflow model specified in modelURL.
        """
        # extract filename and model names from modelURL
        filename = os.path.basename(modelURL)
        self.modelName = filename[:filename.index('.')]

        # define a cache directory where all the model will be stored
        self.cacheDir = "./pretrained_models"

        # create the directory, if not already exist, otherwise skip
        os.makedirs(self.cacheDir, exist_ok=True)

        # download the model
        get_file(fname=filename, origin=modelURL, cache_dir=self.cacheDir, cache_subdir='checkpoints', extract=True)

    def loadModel(self):
        """
        Load downloaded model.
        """
        print('[INFO] Loading Model ' + self.modelName)
        # clear session
        tf.keras.backend.clear_session()
        # load model
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print('[INFO] Model ' + self.modelName + ' is loaded successfully')


    def createBoundingBox(self, image, threshold=0.5):
        """
        Draw bounding box on all predicted objects in the image.
        """
        # convert the cv2 loaded image (in BGR) to RGB, this will gives us numpy array
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        # convert the numpy array to tensor
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        # expand its dimension, since tensorflow take a batch as input, and the input image is only 1 input
        inputTensor = inputTensor[tf.newaxis,...]
        # inputTensor = tf.expand_dims(inputTensor, axis=0)

        # get detection by passing the input to the model, this will gives us a dictionary
        detections = self.model(inputTensor)
        
        # extract bounding boxes from detection dictionary and convert to numpy array
        bboxs = detections['detection_boxes'][0]
        bboxs = np.asarray(bboxs)
        # get the index of the class labels and convert into numpy array of int datatype
        classIndexes = detections['detection_classes'][0]
        classIndexes = np.asarray(classIndexes).astype(np.int32)
        # extract confidence score for each class labels and convert them to numpy array
        classScores = detections['detection_scores'][0]
        classScores = np.asarray(classScores).astype(np.float32)

        # get image height, width and number of channels
        imgH, imgW, imgC = image.shape

        # remove overlapping bounding boxes, this will give the indexes of bounding boxes that meet these criteria
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

        # if there is any bounding boxes, we need to draw each of the bounding boxes
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                # extract 1 bounding box as tuple
                bbox = tuple(bboxs[i].tolist())
                # get the class confidence for the current bounding box
                classConfidence = round(100 * classScores[i])
                # get the class label index
                classIndex = classIndexes[i]

                # get the class name
                classLabelText = self.classesList[classIndex].upper()
                # get the class color
                classColor = self.colorList[classIndex]

                # format the bounding box display text
                displayText = f"{classLabelText}: {classConfidence}%"

                # unpack the bounding box to get the value of the pixels at x and y axis
                # these values are the coordinates of the bounding box relative to the height and width of the image (they are not absolute locations)
                ymin, xmin, ymax, xmax = bbox
                # get the actual location of the pixels, multiplies x axis coord with image width and y axis coord with image height
                xmin, xmax, ymin, ymax = (xmin*imgW, xmax*imgW, ymin*imgH, ymax*imgH)
                # convert the float values into int
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                # plot the bounding box using cv2, pass starting points as pt1 arg; and ending points as pt2 arg
                cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=classColor, thickness=3)
                # display text on the drawn image
                # cv2.putText(image, text=displayText, org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=classColor, thickness=2) # display class label without background
                self.drawTextBorder(image, displayText, pos=(xmin, ymin-10), text_color_bg=classColor) # display class label with background
                
        # return image on which we have drawn bounding boxes
        return image

    def drawTextBorder(self, image, displayText, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=1, font_thickness=2, text_color=(255, 255, 255), text_color_bg=(0, 0, 0)):
        """
        Draw `classColor` filled rectangle for `displayText` background.
        """
        # get position of text
        x, y = pos
        # get text size
        textSize, _ = cv2.getTextSize(displayText, font, font_scale, font_thickness)
        # get width and height of text
        textWdith, textHeight = textSize
        # draw rectangle
        cv2.rectangle(image, pos, (x + textWdith, y + textHeight), text_color_bg, -1)
        # display image
        cv2.putText(image, displayText, (x, y + textHeight + font_scale - 1), font, font_scale, text_color, font_thickness)

    def predictImage(self, imagePath, threshold=0.5):
        """
        Predict all objects on image.
        """
        # read image
        image = cv2.imread(imagePath)
        
        # get image filename with extension
        image_filename = os.path.basename(imagePath)
        # set folder to save output image
        saved_image_folder = os.path.join('test', self.modelName)
        # create folder if not already exist, else, do nothing
        os.makedirs(saved_image_folder, exist_ok=True)
        # set saving path
        new_filename = os.path.join(saved_image_folder, image_filename)

        # create bounding box on image
        bboxImage = self.createBoundingBox(image, threshold)

        # save the image
        if not cv2.imwrite(new_filename, bboxImage):
            raise Exception('[Error] Could not save image')
        # show result
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold=0.5):
        """
        Predict all object on video.
        """
        # capture the video
        cap = cv2.VideoCapture(videoPath)
        frameWidth = int(cap.get(3))
        frameHeight = int(cap.get(4))

        # define saving location
        video_filename = os.path.basename(videoPath)
        saved_image_folder = os.path.join('test', self.modelName)
        os.makedirs(saved_image_folder, exist_ok=True)
        new_filename = os.path.join(saved_image_folder, video_filename)

        # define codec and videoWriter instance
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(new_filename, fourcc, 20.0, (frameWidth,frameHeight))

        # if can't capture video
        if (cap.isOpened() == False):
            print("[ERROR] Unable to open file")
            return
        
        # otherwise read the image
        (success, image) = cap.read()
        startTime = 0

        # while image capture is success
        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime

            # create bounding box on the current frame, which gives us frame with bounding box
            bboxImage = self.createBoundingBox(image, threshold)
            # put fps text on the image
            cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

            # save the frame
            out.write(image)

            # show the video frame
            cv2.imshow("Result", bboxImage)

            # define keypress
            key = cv2.waitKey(1) & 0xFF
            
            # pressing 'q' on the keyboard will break the loop
            if key == ord('q'):
                break

            # otherwise, capture next frame
            (success, image) = cap.read()
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()