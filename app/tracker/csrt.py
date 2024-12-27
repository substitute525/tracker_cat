import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Callable

import cv2
from numpy import ndarray

class InitStrategy(Enum):
    """
    重新标记策略
    """
    BY_SECONDS = 1  # 按秒
    BY_UPDATE = 2   # 按更新次数
    WHEN_FREE = 3   # 当空闲时（可指定最小时间间隔和最大间隔）
    WHEN_LOST = 4   # 当跟踪失败时（可指定最小时间间隔和最大间隔）

class CsrtVideoStream:
    cap = None
    tracker = None
    queue = None
    _reinit_func = None
    wait_init = False
    event = threading.Event()
    threads = []
    update_frames = 0
    _finished = False

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
        self.frame_interval = interval
        self.frames = queue.PriorityQueue()

    def re_init_strategy(self, strategy:InitStrategy, **kwargs):
        """
        重计算策略(多次调用会导致多个策略同时进行)
        :param strategy: 计算策略
        :param kwargs: 策略参数。
        :return:
        """
        if strategy == InitStrategy.BY_SECONDS:
            interval = kwargs.get('interval')
            thread = threading.Thread(target=self._reinit_loop, name="csrtVideoStram-initBySeconds", args=(interval, strategy), daemon=True)
            self.threads.append(thread)
            thread.start()
            ...
        elif strategy == InitStrategy.BY_UPDATE:
            interval = kwargs.get('interval')
            thread = threading.Thread(target=self._reinit_loop, name="csrtVideoStram-initBySeconds", args=(interval, strategy), daemon=True)
            self.threads.append(thread)
            thread.start()
            ...
        elif strategy == InitStrategy.WHEN_FREE:
            min_interval = kwargs.get('min_interval')
            max_interval = kwargs.get('max_interval')
            ...
        elif strategy == InitStrategy.WHEN_LOST:
            min_interval = kwargs.get('min_interval')
            max_interval = kwargs.get('max_interval')
            ...

    def track(self, func: Callable[[ndarray], list[int]]):
        """
        跟踪选定框
        :param func: 计算帧的目标区域
        """
        self._finished = False
        ret, frame = self.cap.read()
        if not ret:
            return
        self._reinit_func = func
        bbox = func(frame)
        if not bbox:
            return
        self.tracker.init(frame, bbox)
        self._track()
        self._finished = True
        self.queue.join()

    def _track(self):
        interval = 0
        start = time.time()
        while True:
            ret, frame = self.cap.read()
            # 抽帧
            if interval != self.frame_interval:
                interval += 1
                continue
            else:
                interval = 0
            # 结束
            if not ret:
                break
            else:
                self.queue.put(frame)
                self.update_frames += 1
                self._pool_executor.submit(self._update, time.time_ns())
            if self.wait_init:
                print("等待重计算")
                self.event.wait()
                self.event.clear()
                print("结束等待")
            # 按 'q' 键退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        print(f"update frame:{self.update_frames}, track time {time.time() - start}")

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

    def next_track_frame(self, blocking=True, timeout=None):
        """
        获取处理后的下一帧（若开启了save_frames）
        :param blocking: 阻塞获取
        :param timeout: 超时时间
        :return: 下一帧
        """
        return self.frames.get(blocking, timeout) if self.save_frames and not self.frames.empty() else (None,None)

    def release(self):
        """
        释放当前跟踪器的资源
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def _reinit_loop(self, interval: int, strategy: InitStrategy):
        start = time.time()
        while True:
            if self._finished:
                print(f"跟踪器已结束，更新策略：{strategy.name}退出")
                return
            if self.cap is None or not self.cap.isOpened():
                print("no cap")
                continue
            if strategy == InitStrategy.BY_SECONDS:
                if (time.time() - start) < interval:
                    continue
            elif strategy == InitStrategy.BY_UPDATE:
                if self.update_frames % interval != 0 or self.update_frames == 0:
                    continue
            print("start reinit")
            # tread
            self.wait_init = True
            # 等待update完成
            self.queue.join()
            ret, init_frame = self.cap.read()
            if not ret:
                self.wait_init = False
                self.event.set()
                continue
            bbox = self._reinit_func(init_frame)
            print("reinit bbox")
            if not bbox:
                print(f"no bbox, current frames:{self.update_frames}")
                self.wait_init = False
                self.event.set()
                continue
            self.tracker.init(init_frame, bbox)
            self.wait_init = False
            self.event.set()
            start = time.time()
            print("end reinit")