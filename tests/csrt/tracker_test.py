import time

import cv2

from app import tracker
from app.model import yolo

if __name__ == '__main__':
    stream = tracker.csrt.CsrtVideoStream(r"D:\video.mp4", True, 3, 1)
    ret, frame = stream.cap.read()
    bbox = tracker.select_box(frame)
    yolo4 = yolo.yolo4.Yolo4()
    model = yolo4.load_model()
    yolo4.
    cv2.destroyAllWindows()
    time_time = time.time()
    stream.track(frame, bbox)
    time_cost = time.time()
    print('track cost:', time_cost - time_time)
    while True:
        _, frame = stream.get_frame()
        if frame is None:
            break
        cv2.imshow("Tracking", frame)
        cv2.waitKey()
    stream.release()
