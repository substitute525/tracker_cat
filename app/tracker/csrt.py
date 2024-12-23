import queue

import cv2
from concurrent.futures import ThreadPoolExecutor

if __name__ == '__main__':
    help(cv2.selectROI)

class CsrtVideoStream:
    def __init__(self, video_source: int|str):
        self.cap = cv2.VideoCapture(video_source)
        self.tracker = cv2.TrackerCSRT_create()
        self._queue = queue.Queue(maxsize=240)


    def track(self, frame, bbox):
        self.tracker.init(frame, bbox)
        self._track()
        with ThreadPoolExecutor(max_workers=60) as executor:
            self._queue.get()
            results = executor.submit(self._update, frame)

    def _track(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            else:
                self._queue.put(frame)
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _update(self, frame):
        ret, bbox = self.tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost Tracking", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Tracking", frame)

    def get_frame(self):
        return self._queue.queue[0]

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
