import time

import cv2

from app import tracker
from app.model import yolo
from app.model import yolo_most_like_box

if __name__ == '__main__':
    stream = tracker.csrt.CsrtVideoStream(r"D:\video.mp4", True, 3, 2)
    total_frame = int(stream.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = stream.cap.get(cv2.CAP_PROP_FPS)
    duration = total_frame / fps

    start = time.time()
    ret, frame = stream.cap.read()
    yolo_ = yolo.yolo4.Yolo4()
    imread, bbox = yolo_most_like_box(yolo_, frame, confidence=0.1, class_id=15)
    if bbox is None:
        print("nothing found")
        exit()
    yolo4 = yolo.yolo4.Yolo4()
    cv2.destroyAllWindows()
    stream.track(frame, bbox)
    print(f'track cost:{time.time() - start}, total frame:{total_frame}, fps:{fps}, video duration:{duration}')
    while True:
        _, frame = stream.next_track_frame()
        if frame is None:
            break
        cv2.imshow("Tracking", frame)
        cv2.waitKey(50)
    stream.release()
