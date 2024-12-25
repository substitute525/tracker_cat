import time

import cv2

from app import tracker
from app.model import yolo
from app.model import yolo_most_like_box

if __name__ == '__main__':
    stream = tracker.csrt.CsrtVideoStream(r"D:\video.mp4", True, 3, 1)
    ret, frame = stream.cap.read()
    # bbox = tracker.select_box(frame)
    yolo_ = yolo.yolo4.Yolo4()
    imread, bbox = yolo_most_like_box(yolo_, frame, 15, 0.1)
    yolo4 = yolo.yolo4.Yolo4()
    cv2.destroyAllWindows()
    time_time = time.time()
    stream.track(frame, bbox)
    time_cost = time.time()
    print('track cost:', time_cost - time_time)
    while True:
        _, frame = stream.next_track_frame()
        if frame is None:
            break
        cv2.imshow("Tracking", frame)
        cv2.waitKey(10)
    stream.release()
