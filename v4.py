import cv2
import sys
from yolo.yololo import Yolo
import time

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


def get_first_frame(file):
    v = cv2.VideoCapture(file)

    if not v.isOpened():
        print("Could not open video")
        sys.exit()

    result, f = v.read()
    if not result:
        print('Cannot read video file')
        sys.exit()

    return f, v


if __name__ == '__main__':
    filename = 'videos/video_2.hevc'
    first_frame, vid = get_first_frame(filename)

    y = Yolo(im_shape=(864., 1152.))

    timer = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    while True:
        ok, img = vid.read()

        if not ok:
            sys.exit("failed reading next frame")

        timer = cv2.getTickCount()
        start = time.time()
        boxes = y.predict(img)
        end = time.time()

        if len(boxes) > 0:
            top, left, bottom, right = boxes[0]

            top = max(0, top)
            left = max(0, left)
            bottom = min(img.shape[0], bottom)
            right = min(img.shape[1], right)

            cv2.rectangle(img, (left, top), (right, bottom), BLUE, 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        cv2.putText(img, "FPS: {}".format(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2)
        cv2.putText(img, "YOLO delay: {:0.2f}s".format(end-start), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2)
        cv2.imshow('img', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
