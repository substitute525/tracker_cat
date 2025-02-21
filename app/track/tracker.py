import queue
import random
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Callable

import cv2
from numpy import ndarray

logger = get_logger("csrt_tracker")

class InitStrategy(Enum):
    """
    重新标记策略
    """
    BY_MILL_SECONDS = 1  # 按毫秒
    BY_UPDATE = 2   # 按更新次数
    WHEN_FREE = 3   # 当空闲时（可指定最小时间间隔和最大间隔）
    WHEN_LOST = 4   # 当跟踪失败时（可指定最小时间间隔和最大间隔）

class ALG(Enum):
    CSRT = "csrt"
    MOSSE = "mosse"
    KCF = "kcf"
    GOTURN = "goturn"
    DaSiamRPN = "dasiamrpn"

class VideoStream:
    cap = None
    tracker = None
    update_frames = 0       # 已更新帧数
    last_init_time = 0      # 上次更新时间
    guarantee_inited = False    # 启动担保
    calc_queue = None       # 计算队列
    wait_init = threading.Lock()    # reinit锁
    condition = threading.Condition()    # lostReinit唤醒

    frame_interval = 0      # 抽帧间隔
    alg = ALG.KCF           # 追踪算法
    frames_buffer = None    # 视频帧缓冲队列
    strategy_threads = []   # 策略线程
    save_frames = False     # 是否保存处理后的帧
    processed_frames = None # 处理后的帧
    lost_min_interval = 1000 # 跟踪失败初始化间隔
    lost_reinit = False      # 跟踪失败初始化
    screen_width = None     # 屏幕宽度
    screen_height = None     # 屏幕高度

    _finished = False       # 视频是否读取完成
    calc_finished = False   # 视频是否计算完成
    _pool_executor = None   # 计算线程池
    _reinit_func = None     # 重框选方法

    def signal_handler(self):
        self._finished = True
        self.calc_finished = True

    def __init__(self, video_source: int | str, **kwargs):
        """
        初始化CSRT追踪算法

        :param video_source: 视频源
        :param save_frames: 是否保存帧
        :param interval: 抽帧间隔，0不抽帧
        :param parallelism: 并行度，仅推荐1
        """
        signal.signal(signal.SIGINT, self.signal_handler)
        self.cap = cv2.VideoCapture(video_source)
        self.alg = kwargs.get("alg", ALG.KCF)
        self.save_frames = kwargs.get("save_frames", False)
        resize = kwargs.get("resize", [None, None])
        if isinstance(resize, (list, tuple)):
            if len(resize) == 2:
                self.screen_width, self.screen_height = resize
            elif len(resize) == 1:
                self.screen_width = self.screen_height = resize[0]
        parallelism = kwargs.get("parallelism", 1)
        self.frame_interval = kwargs.get("interval", 0)
        calc_queue_size = kwargs.get("calc_queue_size", parallelism * 2 if parallelism > 1 else 0)
        self.calc_queue = queue.Queue(maxsize=(calc_queue_size))
        self._pool_executor = ThreadPoolExecutor(max_workers=parallelism)
        self.create_tracker()
        logger.info(f"{self.alg} tracker initialized")
        self.frames_buffer = queue.Queue(maxsize=kwargs.get("frames_buffer_size", 100))
        if self.save_frames:
            self.processed_frames = queue.PriorityQueue()
        self.position = queue.PriorityQueue()

    def create_tracker(self):
        if self.alg == ALG.MOSSE:
            create = cv2.legacy.TrackerMOSSE_create()
            self.tracker = create
        elif self.alg == ALG.KCF:
            # opencv4.5.5的kcf算法多次init存在问题，在4.9版本中解决，因此此处任然使用legacy下的算法
            self.tracker = cv2.legacy.TrackerKCF_create()
        elif self.alg == ALG.CSRT and self.tracker is None:
            self.tracker = cv2.TrackerCSRT_create()
        elif self.alg == ALG.GOTURN and self.tracker is None:
            self.tracker = cv2.TrackerGOTURN_create()
        elif self.alg == ALG.DaSiamRPN and self.tracker is None:
            self.tracker = cv2.TrackerDaSiamRPN_create()

    def re_init_strategy(self, strategy:InitStrategy, **kwargs):
        """
        重计算策略(多次调用会导致多个策略同时进行)
        :param strategy: 计算策略
        :param kwargs: 策略参数。最小间隔控制在100ms以上，否则可能出现线程冲突
        :return:
        """
        logger.info(f"add reinit strategy {strategy}, kwargs={kwargs}")
        if strategy == InitStrategy.BY_MILL_SECONDS:
            interval = kwargs.get('interval')
            thread = threading.Thread(target=self._reinit_loop, name="csrtVideoStram-initByMillSeconds", args=(interval, strategy), daemon=True)
            self.strategy_threads.append(thread)
        elif strategy == InitStrategy.BY_UPDATE:
            interval = kwargs.get('interval')
            thread = threading.Thread(target=self._reinit_loop, name="csrtVideoStram-initByUpdate", args=(interval, strategy), daemon=True)
            self.strategy_threads.append(thread)
        elif strategy == InitStrategy.WHEN_FREE:
            min_interval = kwargs.get('min_interval')
            if min_interval is None:
                min_interval = 100
            max_interval = kwargs.get('max_interval')
            if min_interval is None:
                min_interval = 0
            thread1 = threading.Thread(target=self._reinit_loop, name="csrtVideoStram-initByFree", args=(max_interval, strategy), daemon=True)
            thread2 = threading.Thread(target=self._reinit_free, name="csrtVideoStram-initWhenFree", args=([min_interval]), daemon=True)
            self.strategy_threads.append(thread1)
            self.strategy_threads.append(thread2)
        elif strategy == InitStrategy.WHEN_LOST:
            min_interval = kwargs.get('min_interval')
            if min_interval is None:
                min_interval = 100
            max_interval = kwargs.get('max_interval')
            if min_interval is None:
                min_interval = 0
            self.lost_min_interval = min_interval
            self.lost_reinit = True
            thread1 = threading.Thread(target=self._reinit_loop, name="csrtVideoStram-initByFree", args=(max_interval, strategy), daemon=True)
            thread2 = threading.Thread(target=self._reinit_lost, name="csrtVideoStram-initByLost", daemon=True)
            self.strategy_threads.append(thread1)
            self.strategy_threads.append(thread2)

    def _put_frame(self):
        logger.info("begin read_frame")
        interval = 0
        while True:
            if self._finished:
                break
            if not self.cap.isOpened():
                logger.info("Error: Cannot access the camera.")
                self._finished = True
                break
            ret, frame = self.cap.read()
            if not ret:
                logger.info("end of read frame")
                self._finished = True
                break
            if frame is None:
                logger.info("frame is None")
            # 抽帧
            if interval != self.frame_interval:
                interval += 1
                continue
            else:
                interval = 0
            # 缩放
            if (self.screen_width is not None  and self.screen_height is not None
                    and (frame.shape[1] > self.screen_width or frame.shape[0] > self.screen_height)):
                scale_width = self.screen_width / frame.shape[1]
                scale_height = self.screen_height / frame.shape[0]
                scale = min(scale_width, scale_height)
                width = int(frame.shape[1] * scale)
                height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            self.frames_buffer.put(frame)

    def _read_frame(self):
        while True:
            if self._finished and self.frames_buffer.empty():
               return None
            try:
                frame = self.frames_buffer.get(timeout=0.1)
                if frame is not None:
                    return frame
            except queue.Empty:
                continue

    def _read_last_frame(self):
        while True:
            if self._finished or self.frames_buffer.empty():
               return None
            try:
                frame = self.frames_buffer.queue[-1]
                self.frames_buffer = queue.Queue(maxsize = self.frames_buffer.maxsize)
                return frame
            except queue.Empty:
                continue

    def track(self, func: Callable[[ndarray], list[int]]):
        """
        跟踪选定框
        :param func: 计算帧的目标区域
        """
        self._finished = False
        threading.Thread(target=self._put_frame, name="csrtVideoStram-readFrame", daemon=True).start()
        while True:
            frame = self._read_last_frame()
            self._reinit_func = func
            if self._finished:
                logger.info("no frame")
                return
            if frame is None:
                continue
            bbox = func(frame)
            if not bbox:
                continue
            else:
                break
        self.tracker.init(frame, bbox)
        if self.strategy_threads and self.strategy_threads.__len__() > 0:
            logger.info(f"tracker begin, have {self.strategy_threads.__len__()} re_init_strategy")
            for thread in self.strategy_threads:
                thread.start()
        self._track()
        self.calc_queue.join()
        logger.info(f"done, waitsize {self.frames_buffer.qsize()}, update {self.update_frames} frames")

    def _track(self):
        self._pool_executor.submit(self._update)
        while True:
            frame = self._read_frame()
            # 结束
            if frame is None:
                break
            else:
                self.calc_queue.put(frame)
                self.update_frames += 1
            if self.wait_init.locked():
                logger.debug("waiting for reinit")
                # 此处获取锁可能导致reinit跳过
                with self.wait_init:
                    logger.debug("continue update")
            # 按 'q' 键退出
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    def _update(self):
        priority = 0
        while True:
            if self.frames_buffer.empty() and self._finished and self.calc_queue.empty():
                self.calc_finished = True
                return
            time_ns = time.time_ns()
            try:
                frame = self.calc_queue.get(timeout=10)
                ret, bbox = self.tracker.update(frame)
                if ret:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                    self.position.put(((time_ns, self.update_frames, priority), center))
                    if self.save_frames:
                        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                        self.processed_frames.put(((time_ns, self.update_frames, priority), frame))
                else:
                    self.position.put(((time_ns, self.update_frames, priority), None))
                    if self.save_frames:
                        cv2.putText(frame, "Lost Tracking", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        self.processed_frames.put(((time_ns, self.update_frames, priority), frame))
                # 记录所有帧
                priority = priority + 1
                if priority > 2**31 - 1:
                    priority = 0
                if not ret:
                    # 是否执行LostReInit策略
                    with self.condition:
                        self.condition.notify()
                self.calc_queue.task_done()
                ...
            except Exception as e:
                logger.error(e)

    def next_track_frame(self, blocking=True, timeout=None):
        """
        获取处理后的下一帧（若开启了save_frames）
        :param blocking: 阻塞获取
        :param timeout: 超时时间
        :return: 下一帧
        """
        return self.processed_frames.get(blocking, timeout) if self.save_frames and not self.processed_frames.empty() else (None, None)

    def next_position(self, blocking=True, timeout=None):
        """
        获取处理后的下一帧（若开启了save_frames）
        :param blocking: 阻塞获取
        :param timeout: 超时时间
        :return: 下一帧
        """
        return self.position.get(blocking, timeout) if not self.processed_frames.empty() else (None, None)

    def release(self):
        """
        释放当前跟踪器的资源
        """
        self.cap.release()

    def _reinit_loop(self, interval: int, strategy: InitStrategy):
        """
        循环更新策略
        :param interval: 更新间隔
        :param strategy: 更新策略
        """
        if not interval or interval == 0 :
            logger.info(f"tracker reinit strategy:{strategy.name} has no interval, quit")
            return
        start = int(time.time() * 1000)
        logger.info(f"tracker reinit strategy:{strategy.name} start")
        def continue_loop():
            nonlocal start
            start = int(time.time() * 1000)

        while True:
            if self.frames_buffer.empty() and self._finished:
                logger.debug(f"main tracker finished,strategy:{strategy.name} quit")
                break
            if self.cap is None or not self.cap.isOpened():
                logger.debug("no cap")
                continue
            if strategy == InitStrategy.BY_MILL_SECONDS:
                if (int(time.time() * 1000) - start) < interval:
                    continue
            elif strategy == InitStrategy.BY_UPDATE:
                if self.update_frames % interval != 0 or self.update_frames == 0:
                    continue
            elif strategy == InitStrategy.WHEN_FREE:
                now = int(time.time() * 1000)
                if not self.last_init_time :
                    self.last_init_time = now
                if now - self.last_init_time > interval:
                    logger.debug(f"No idle state in {interval} milliseconds, guaranteed init starts")
                else:
                    continue
            logger.debug(f"trigger reinit, queue size {self.calc_queue.qsize()}, unfinished tasks:{self.calc_queue.unfinished_tasks}")
            trigger = int(time.time() * 1000)
            # tread
            if not self.wait_init.acquire(blocking=False):
                logger.debug("reinit triggered by other thread")
                start = int(time.time() * 1000)
                continue
            try:
                self.last_init_time = int(time.time() * 1000)
                # 等待update完成
                self.calc_queue.join()
                logger.debug("waiting reinit finished")
                init_frame = self._read_frame()
                if init_frame is None:
                    logger.debug(f"no frame,continue")
                    continue_loop()
                    continue
                bbox = self._reinit_func(init_frame)
                if not bbox:
                    logger.debug(f"found none bbox, current frames:{self.update_frames}")
                    continue_loop()
                    continue
                self.create_tracker()
                self.tracker.init(init_frame, bbox)
                continue_loop()
                logger.debug(f"reinit cost {int(time.time() * 1000) - trigger}ms")
            finally:
                self.wait_init.release()
        logger.info(f"tracker reinit strategy:{strategy.name} quit")

    def _reinit_free(self, min_interval: int):
        if not self.last_init_time:
            self.last_init_time = int(time.time() * 1000)
        while True:
            self.calc_queue.join()
            if self.frames_buffer.empty() and self._finished:
                logger.info(f"main tracker finished, quit")
                break
            if int(time.time() * 1000) - self.last_init_time <= min_interval or self.wait_init.locked():
                continue
            if not self.wait_init.acquire(blocking=False):
                logger.debug("reinit triggered by other thread")
                continue
            try:
                self.last_init_time = int(time.time() * 1000)
                logger.info("trigger reinit by strategy: WHEN_FREE")
                self.calc_queue.join()
                trigger = int(time.time() * 1000)
                init_frame = self._read_frame()
                if init_frame is None:
                    logger.debug(f"no frame,continue")
                    continue
                bbox = self._reinit_func(init_frame)
                if not bbox:
                    logger.debug(f"found none bbox, current frames:{self.update_frames}")
                    continue
                self.create_tracker()
                self.tracker.init(init_frame, bbox)
                logger.debug(f"reinit cost {int(time.time() * 1000) - trigger}ms")
            finally:
                self.wait_init.release()

    def _reinit_lost(self):
        while True:
            if not self.lost_reinit:
                return
            with self.condition:
                self.condition.wait()
            if not self.last_init_time:
                self.last_init_time = int(time.time() * 1000)
            if self.frames_buffer.empty() and self.calc_finished:
                logger.info(f"main tracker finished, quit")
                return
            if int(time.time() * 1000) - self.last_init_time <= self.lost_min_interval or self.wait_init.locked():
                continue
            if not self.wait_init.acquire(blocking=False):
                logger.debug("reinit triggered by other thread")
                continue
            self.calc_queue.join()
            try:
                self.last_init_time = int(time.time() * 1000)
                logger.info("trigger reinit by strategy: WHEN_LOST")
                self.calc_queue.join()
                trigger = int(time.time() * 1000)
                init_frame = self._read_frame()
                if init_frame is None:
                    logger.debug(f"no frame,continue")
                    continue
                bbox = self._reinit_func(init_frame)
                if not bbox:
                    logger.debug(f"found none bbox, current frames:{self.update_frames}")
                    continue
                self.create_tracker()
                self.tracker.init(init_frame, bbox)
                logger.debug(f"reinit cost {int(time.time() * 1000) - trigger}ms")
            finally:
                self.wait_init.release()