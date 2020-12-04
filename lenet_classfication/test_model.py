# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import glob
import time

model = load_model("model_number.h5")

for filename in glob.glob('test/*.jpg'):
    # filename =r"sample_image.png"
    # cap = cv2.VideoCapture(0)
    frame = cv2.imread(filename)
    image = cv2.resize(frame, (64, 64))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    predict = model.predict(image)[0]

    label = np.argmax(predict)
    label = str(label)
    frame = cv2.resize(frame, (300, 300))
    cv2.putText(frame, label,(50, 50),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0),2)
    cv2.imshow("image",frame)
    cv2.waitKey(0)
