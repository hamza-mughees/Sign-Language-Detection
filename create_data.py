import os
from string import ascii_uppercase
import cv2

data_dir = 'data'
train_dir = 'train'
test_dir = 'test'

train_path = f'./{data_dir}/{train_dir}'
test_path = f'./{data_dir}/{test_dir}'

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

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    raise IOError('Cannot open camera')

while True:
    ret, frame = cam.read()

    cv2.imshow("", frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cam.release()
cv2.destroyAllWindows()