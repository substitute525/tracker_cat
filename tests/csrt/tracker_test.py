from app import config
config.init_config()

import time

import cv2

from app import tracker
from app.model import yolo
from app.model import yolo_most_like_box
from app.tracker.csrt import InitStrategy

if __name__ == '__main__':
    stream = tracker.csrt.CsrtVideoStream(r"D:\video.mp4", True, 3, 1)
    total_frame = int(stream.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = stream.cap.get(cv2.CAP_PROP_FPS)
    duration = total_frame / fps
    start = time.time()
    yolo_ = yolo.yolo4.Yolo4()
    cv2.destroyAllWindows()
    stream.re_init_strategy(strategy=InitStrategy.BY_UPDATE, interval=20)
    stream.track(lambda frm: yolo_most_like_box(yolo_, frm, confidence=0.1, class_id=15)[1])
    print(f'track cost:{time.time() - start}, total frame:{total_frame}, fps:{fps}, video duration:{duration}')
    while True:
        _, frame = stream.next_track_frame()
        if frame is None:
            break
        cv2.imshow("Tracking", frame)
        cv2.waitKey(50)
    stream.release()
