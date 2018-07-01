import sys
import cv2

faceCascPath = "cascades/haarcascade_frontalface_default.xml"
eyeCascPath = "cascades/haarcascade_eye.xml"
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def get_video():
    v = cv2.VideoCapture("videos/video_5.mp4")

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

    firstFrame, video = get_video()

    while True:
        ret, img = video.read()

        if not ret:
            break

        timer = cv2.getTickCount()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.05, minNeighbors=7, minSize=(80, 80))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), BLUE, 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+w, x:x+h]
            eyes = eyeCascade.detectMultiScale(roi_gray, minNeighbors=7)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), GREEN, 2)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        cv2.putText(img, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2)
        cv2.imshow('img', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    video.release()
    cv2.destroyAllWindows()
