import os
from string import ascii_uppercase
import cv2

resc_dir = 'resources'

data_dir = 'data'
train_dir = 'train'
test_dir = 'test'

train_path = f'./{data_dir}/{train_dir}'
test_path = f'./{data_dir}/{test_dir}'

bg_path = f'./{resc_dir}/bg.png'

if not os.path.exists(resc_dir):
    os.mkdir(resc_dir)

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(train_path):
    os.mkdir(train_path)

if not os.path.exists(test_path):
    os.mkdir(test_path)

for c in ascii_uppercase:
    label_train_path = f'{train_path}/{c}'
    label_test_path = f'{test_path}/{c}'

    if not os.path.exists(label_train_path):
        os.mkdir(label_train_path)
    
    if not os.path.exists(label_test_path):
        os.mkdir(label_test_path)

class Rect_ROI:
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y

        self.w = w
        self.h = h

        self.x2 = x + w
        self.y2 = y + h

rect_roi = Rect_ROI(100, 100, 200, 200)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    raise IOError('Cannot open camera')

while True:
    ret, frame = cam.read()

    # cv2.imshow("", frame)

    roi = frame[rect_roi.y1 : rect_roi.y2, rect_roi.x1 : rect_roi.x2]

    cv2.imshow('Live', roi)

    if os.path.exists(bg_path):
        bg = cv2.imread(bg_path)

        diff = cv2.absdiff(roi, bg)
        cv2.imshow('Difference', diff)

        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, thr = cv2.threshold(diff_gray, 50, 255, cv2.THRESH_BINARY)
        cv2.imshow('Difference Thresholded', thr)

    c = cv2.waitKey(1)

    if c == 27:     # exit
        break
    if c == 98:     # create background image
        cv2.imwrite(bg_path, roi)
    if c == 99:     # capture and save data image
        print('Window captured!')

cam.release()
cv2.destroyAllWindows()