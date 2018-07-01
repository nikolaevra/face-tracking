import cv2
import sys


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
    videos = [2]

    for video in videos:
        filename = '../videos/video_{}.mp4'.format(video)
        first_frame, vid = get_first_frame(filename)
        count = 0
        interval = 1

        eyeCascPath = "../cascades/haarcascade_eye.xml"
        eyeCascade = cv2.CascadeClassifier(eyeCascPath)

        while True:
            ok, frame = vid.read()

            if not ok:
                break

            if count % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eyes = eyeCascade.detectMultiScale(gray, minNeighbors=5)

                for i, (ex, ey, ew, eh) in enumerate(eyes):
                    eye = frame[ey:ey + 50, ex:ex + 50]
                    cv2.imwrite("./data/video_{}_frame_{}_eye_{}.png".format(video, count, i), eye)
                    print("./data/video_{}_frame_{}_eye_{}.png".format(video, count, i))

            count += 1
