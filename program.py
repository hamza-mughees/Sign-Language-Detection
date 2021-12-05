from json import load
from keras.models import load_model
import cv2
import os
import numpy as np
from tensorflow import convert_to_tensor, reshape

resc_dir = 'resources'
bg_path = f'./{resc_dir}/bg.png'

model_id = '1638308364'
model_path = f'./models/{model_id}.h5py'

model = load_model(model_path)

bg_set = False

font = cv2.FONT_HERSHEY_SIMPLEX

x = 100
y = 100

dxy = 5

class Rect_ROI:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x1 = x
        self.y1 = y

        self.w = w
        self.h = h

        self.x2 = x + w
        self.y2 = y + h

rect_roi = Rect_ROI(x, y, 200, 200)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    raise IOError('Cannot open camera')

while True:
    ret, frame = cam.read()

    roi = frame[rect_roi.y1 : rect_roi.y2, rect_roi.x1 : rect_roi.x2]
    frame = cv2.rectangle(frame, (rect_roi.x1, rect_roi.y1), (rect_roi.x2, rect_roi.y2), (0, 255, 0), 2)

    # cv2.imshow('Live', roi)

    if bg_set:
        bg = cv2.imread(bg_path)

        diff = cv2.absdiff(roi, bg)

        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, thr = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Difference Thresholded', thr)

        x = reshape(convert_to_tensor(thr), [1, 200, 200, 1])
        y = np.argmax(model.predict(x))

        if np.sum(x) > 1000000:
            cv2.putText(frame, str(y), (rect_roi.x1 + 10, rect_roi.y1 + 30), font, 1, (0, 255, 0), 1, cv2.LINE_4)
    else:
        pass
        # TODO: Add instructions for the user to use w,a,s,d and b

    cv2.imshow('Live', frame)
    
    c = cv2.waitKey(1)

    if c == 27:                                 # exit
        break

    if not bg_set and c == ord('w'):
        y -= dxy
        if y < 0:
            y += dxy
        rect_roi = Rect_ROI(x, y, 200, 200)

    if not bg_set and c == ord('s'):
        y += dxy
        if y > frame.shape[0] - 200:
            y -= dxy
        rect_roi = Rect_ROI(x, y, 200, 200)

    if not bg_set and c == ord('a'):
        x -= dxy
        if x < 0:
            x += dxy
        rect_roi = Rect_ROI(x, y, 200, 200)

    if not bg_set and c == ord('d'):
        x += dxy
        if x > frame.shape[1] - 200:
            x -= dxy
        rect_roi = Rect_ROI(x, y, 200, 200)

    if c == ord('b'):                           # create background image
        cv2.imwrite(bg_path, roi)
        bg_set = True