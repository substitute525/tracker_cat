import queue
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

if __name__ == '__main__':
    help(cv2.selectROI)


class CsrtVideoStream:
    def __init__(self, video_source: int | str, save_frames: bool = False, interval: int=0, parallelism: int=1):
        """
        初始化CSRT追踪算法

        :param video_source: 视频源
        :param save_frames: 是否保存帧
        :param interval: 抽帧间隔，0不抽帧
        :param parallelism: 并行度，不建议大于2
        """
        self.cap = cv2.VideoCapture(video_source)
        self.tracker = cv2.TrackerCSRT_create()
        self.queue = queue.Queue(maxsize=(parallelism if parallelism >= 1 else 0))
        self._pool_executor = ThreadPoolExecutor(max_workers=parallelism)
        self.save_frames = save_frames
        self.interval = interval
        self.frames = queue.PriorityQueue()


    def track(self, frame, bbox):
        self.tracker.init(frame, bbox)
        self._track()
        self.queue.join()

    def _track(self):
        interval = count = 0
        time_diff = 0
        while True:
            time_time = time.time()
            ret, frame = self.cap.read()
            time_diff += time.time() - time_time
            # 抽帧
            if interval != self.interval:
                interval += 1
                continue
            else:
                interval = 0
            # 结束
            if not ret:
                break
            else:
                self.queue.put(frame)
                count += 1
                self._pool_executor.submit(self._update, time.time_ns())
                print(f"frame {count}")
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(f"time {time_diff}")

    def _update(self, time_ns):
        frame = self.queue.get(timeout=1)
        ret, bbox = self.tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost Tracking", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # cv2.imshow("Tracking", frame)
        # 记录所有帧
        if self.save_frames:
            self.frames.put((time_ns, frame))
        self.queue.task_done()

    def get_frame(self, blocking=True, timeout=None):
        return self.frames.get(blocking, timeout) if self.save_frames and not self.frames.empty() else (None,None)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
