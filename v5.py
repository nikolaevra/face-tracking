import cv2
import sys
from yolo.yololo import Yolo
from eye_classifier.model.eye_opener import EyeOpener
import time

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
BOUNDARIES_OFFSET = {"l": 150, "r": 150, "t": 50, "b": 200}
BLUE = (50, 170, 50)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
IMG_SIZE = 48
_UPDATE = 10


def get_tracker():
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    t_type = tracker_types[2]
    t = None

    if int(minor_ver) < 3:
        t = cv2.Tracker_create(t_type)
    else:
        if t_type == 'BOOSTING':
            t = cv2.TrackerBoosting_create()
        if t_type == 'MIL':
            t = cv2.TrackerMIL_create()
        if t_type == 'KCF':
            t = cv2.TrackerKCF_create()
        if t_type == 'TLD':
            t = cv2.TrackerTLD_create()
        if t_type == 'MEDIANFLOW':
            t = cv2.TrackerMedianFlow_create()
        if t_type == 'GOTURN':
            t = cv2.TrackerGOTURN_create()

    return t, t_type


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
    filename = 'videos/video_5.mp4'
    first_frame, vid = get_first_frame(filename)

    eyeCascPath = "cascades/haarcascade_eye.xml"
    eyeCascade = cv2.CascadeClassifier(eyeCascPath)

    y = Yolo(im_shape=(864., 1152.))
    e = EyeOpener()
    count = 0
    top = bottom = right = left = 0
    image = None

    # Finding person to track
    while True:
        ok, image = vid.read()

        if not ok:
            sys.exit("failed reading next frame")

        start = time.time()
        boxes = y.predict(image)
        end = time.time()
        print("frame", count, "prediction time: ", end - start)

        if len(boxes) > 0:
            top, left, bottom, right = boxes[0]
            top = max(0, top)
            left = max(0, left)
            bottom = min(image.shape[0], bottom)
            right = min(image.shape[1], right)
            break

    tracker, tracker_type = get_tracker()

    # adjusting frame size to speed up tracking and because Yolo detects a human instead of just face
    ok = tracker.init(
        image,
        (
            left + BOUNDARIES_OFFSET["l"],
            top + BOUNDARIES_OFFSET["t"],
            right - left - BOUNDARIES_OFFSET["r"],
            bottom - top - BOUNDARIES_OFFSET["b"]
        )
    )

    if not ok:
        sys.exit("Failed to initialize tracker")

    while True:
        ok, frame = vid.read()
        eyes_open = True

        if not ok:
            sys.exit("Failed to read next frame")

        # calculate FPS and update tracker position
        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            # draw person box
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            # find eyes and draw eye box
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + w, x:x + h]
            eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5)

            for (ex, ey, ew, eh) in eyes:
                eye = e.predict(frame[ey:ey + IMG_SIZE, ex:ex + IMG_SIZE])

                eyes_open = False if eye == "Closed" else True

                if eye != "Other":
                    print(eye)
                    cv2.rectangle(roi_color, (ex, ey), (ex + IMG_SIZE, ey + IMG_SIZE), GREEN, 2)

        else:
            cv2.putText(frame, "Tracking failure detected", (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)

        # show text if eyes are open or closed
        if eyes_open == True:
            cv2.putText(frame, "Eyes Open", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)
        else:
            cv2.putText(frame, "Eyes Closed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLUE, 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLUE, 2)
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
