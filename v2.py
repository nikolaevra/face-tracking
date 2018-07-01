import cv2
import sys

faceCascPath = "cascades/haarcascade_frontalface_default.xml"
eyeCascPath = "cascades/haarcascade_eye.xml"
(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


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


def detect(img, cascade):
    locations = cascade.detectMultiScale(
        img,
        scaleFactor=1.01,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(300, 300),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(locations) == 0:
        return []

    locations[:, 2:] += locations[:, :2]

    return locations


def draw_rects(img, squares, color):
    for x1, y1, x2, y2 in squares:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def get_first_frame():
    v = cv2.VideoCapture("videos/video_2.hevc")

    if not v.isOpened():
        print("Could not open video")
        sys.exit()

    result, f = v.read()
    if not result:
        print('Cannot read video file')
        sys.exit()

    return f, v


if __name__ == '__main__':
    faceCascade = cv2.CascadeClassifier(faceCascPath)
    eyeCascade = cv2.CascadeClassifier(eyeCascPath)

    frame, video = get_first_frame()
    tracker, tracker_type = get_tracker()
    rect = list()

    # detect the face for the first time
    while len(rect) == 0:
        rect = detect(frame, faceCascade)
        ok, frame = video.read()
        if not ok:
            print("failed to read video")
            sys.exit()

    # initialize tracker on face
    tracker.init(frame, (rect[0][0], rect[0][1], rect[0][2] - rect[0][0], rect[0][3] - rect[0][1]))

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ok:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+w, x:x+h]
            eyes = eyeCascade.detectMultiScale(roi_gray, minNeighbors=5)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), GREEN, 2)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2)

        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2)
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
