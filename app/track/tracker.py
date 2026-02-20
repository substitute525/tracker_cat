import sys
import threading
import time
import traceback
from enum import Enum
from typing import Callable, Tuple, Optional
import queue

import cv2
import numpy as np

from app.config.logger import get_logger

logger = get_logger("realtime_tracker")


class InitStrategy(Enum):
    """
    重新标记策略
    """
    BY_MILL_SECONDS = 1  # 按毫秒
    BY_UPDATE = 2  # 按更新次数
    WHEN_FREE = 3  # 当空闲时
    WHEN_LOST = 4  # 当跟踪失败时


class ALG(Enum):
    CSRT = "csrt"
    MOSSE = "mosse"
    KCF = "kcf"
    GOTURN = "goturn"
    DaSiamRPN = "dasiamrpn"


class RealTimeTracker:
    def __init__(self, video_source, alg: ALG = ALG.KCF, **kwargs):
        """
        实时追踪器 - 基于高效双线程架构
        
        :param video_source: 视频源
        :param alg: 追踪算法类型
        :param kwargs: 其他配置参数
        """
        self.video_source = video_source
        self.alg = alg
        self.cap = cv2.VideoCapture(video_source)

        # 实时帧缓存 - 只保留最新帧
        self.latest_frame = None
        self.latest_processed_frame = None
        self.current_bbox = None
        self.tracking_position = None

        # 线程同步
        self.frame_lock = threading.Lock()
        self.tracker_lock = threading.RLock()
        self.running = True

        # 性能监控
        self.frame_count = 0
        self.process_count = 0
        self.last_process_time = 0

        # 重初始化相关
        self.reinit_func = None
        self.last_init_time = 0
        self.min_reinit_interval = kwargs.get('min_reinit_interval', 5)

        # 策略线程
        self.strategy_threads = []
        self.lost_condition = threading.Condition()

        # 初始化追踪器
        self.tracker = self._create_tracker()

        logger.info(f"RealTimeTracker initialized with {alg.value} algorithm")

    def _create_tracker(self):
        """创建追踪器实例"""
        if self.alg == ALG.MOSSE:
            return cv2.legacy.TrackerMOSSE_create()
        elif self.alg == ALG.KCF:
            return cv2.legacy.TrackerKCF_create()
        elif self.alg == ALG.CSRT:
            return cv2.TrackerCSRT_create()
        elif self.alg == ALG.GOTURN:
            return cv2.TrackerGOTURN_create()
        elif self.alg == ALG.DaSiamRPN:
            return cv2.TrackerDaSiamRPN_create()
        else:
            return cv2.legacy.TrackerKCF_create()  # 默认使用KCF

    def _frame_producer(self):
        """帧生产者线程 - 实时读取视频帧"""
        logger.info("Frame producer started")
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.debug("No frame was obtained. This might be because the camera is not turned on.")
                    continue

                # 直接覆盖最新帧，丢弃旧帧
                self.latest_frame = [frame.copy(), self.frame_count]
                self.frame_count += 1
                if self.frame_count > 10000000:
                    self.frame_count = 0

            except Exception as e:
                logger.error(f"Frame producer error: {e}")
                break

        logger.info("Frame producer stopped")

    def _tracking_processor(self):
        """追踪处理线程 - 实时处理最新帧"""
        logger.info("Tracking processor started")
        while self.running:
            time.sleep(0.01)  # 10ms轮询
            try:
                # 获取最新帧
                current_frame = None
                index = 0
                with self.frame_lock:
                    if self.latest_frame is not None:
                        current_frame = self.latest_frame[0].copy()
                        index = self.latest_frame[1]

                if current_frame is None:
                    continue
                if self.current_bbox is None:
                    # 更新处理后的帧
                    cv2.putText(current_frame, "Lost Tracking",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)
                    self.latest_processed_frame = [current_frame, index]
                    self.process_count += 1
                    self.last_process_time = time.time()
                    # 触发丢失重初始化
                    with self.lost_condition:
                        self.lost_condition.notify_all()
                    continue
                # 追踪处理
                with self.tracker_lock:
                    if self.current_bbox is not None and self.tracker is not None:
                        success, bbox = self.tracker.update(current_frame)
                        if self.latest_processed_frame is not None:
                            if self.latest_processed_frame[1] > index and index < self.frame_count:
                                continue  # 说明已经是老数据了

                        if success:
                            self.current_bbox = bbox
                            # 计算中心点
                            center_x = int(bbox[0] + bbox[2] / 2)
                            center_y = int(bbox[1] + bbox[3] / 2)
                            self.tracking_position = [(center_x, center_y), 0]

                            # 绘制结果
                            p1 = (int(bbox[0]), int(bbox[1]))
                            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                            cv2.rectangle(current_frame, p1, p2, (0, 255, 0), 2)
                            cv2.circle(current_frame, self.tracking_position[0], 5, (0, 0, 255), -1)
                        else:
                            # 追踪失败
                            self.current_bbox = None
                            self.tracking_position = None
                            cv2.putText(current_frame, "Lost Tracking",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2)

                            # 触发丢失重初始化
                            with self.lost_condition:
                                self.lost_condition.notify_all()

                    # 更新处理后的帧
                    self.latest_processed_frame = [current_frame, index]
                    self.process_count += 1
                    self.last_process_time = time.time()
            except Exception as e:
                logger.error(f"Tracking processor error: {e}")
                time.sleep(0.01)

        logger.info("Tracking processor stopped")

    def add_reinit_strategy(self, strategy: InitStrategy, **kwargs):
        """添加重初始化策略"""
        logger.info(f"Adding reinit strategy: {strategy.name}")

        if strategy == InitStrategy.BY_MILL_SECONDS:
            interval = kwargs.get('interval', 5)  # 默认5秒
            thread = threading.Thread(
                target=self._reinit_by_time,
                args=(interval,),
                name="ReinitByTime",
                daemon=True
            )
            self.strategy_threads.append(thread)

        elif strategy == InitStrategy.BY_UPDATE:
            interval = kwargs.get('interval', 100)  # 默认每100帧
            thread = threading.Thread(
                target=self._reinit_by_update,
                args=(interval,),
                name="ReinitByUpdate",
                daemon=True
            )
            self.strategy_threads.append(thread)

        elif strategy == InitStrategy.WHEN_LOST:
            min_interval = kwargs.get('min_interval', 10)
            thread = threading.Thread(
                target=self._reinit_when_lost,
                args=(min_interval,),
                name="ReinitWhenLost",
                daemon=True
            )
            self.strategy_threads.append(thread)

        elif strategy == InitStrategy.WHEN_FREE:
            min_interval = kwargs.get('min_interval', 10)
            thread = threading.Thread(
                target=self._reinit_when_free,
                args=(min_interval,),
                name="ReinitWhenFree",
                daemon=True
            )
            self.strategy_threads.append(thread)

    def _reinit_by_time(self, interval: int):
        """按时间间隔重初始化"""
        while self.running:
            time.sleep(interval)
            if not self.running:
                break
            self._request_reinit()

    def _reinit_by_update(self, interval_frames: int):
        """按帧数间隔重初始化"""
        last_reinit_frame = 0
        while self.running:
            if self.frame_count - last_reinit_frame >= interval_frames:
                self._request_reinit()
                last_reinit_frame = self.frame_count
            time.sleep(0.01)  # 10ms检查间隔

    def _reinit_when_lost(self, min_interval_ms: int):
        """当追踪丢失时重初始化"""
        while self.running:
            with self.lost_condition:
                self.lost_condition.wait()  # 等待丢失信号
                if (time.time() - self.last_init_time > min_interval_ms and
                        self.running and self.reinit_func):
                    self._request_reinit()
                    time.sleep(0.5)  # 500ms检查间隔

    def _reinit_when_free(self, min_interval_ms: int):
        """当系统空闲时重初始化"""
        while self.running:
            # 检查是否有足够的时间间隔且没有正在进行的处理
            current_time = time.time()
            if (current_time - self.last_init_time > min_interval_ms and
                    self.process_count == self.frame_count):
                self._request_reinit()
            time.sleep(0.1)  # 100ms检查间隔

    def _request_reinit(self):
        """请求重初始化"""
        if not self.reinit_func or not self.running:
            return
        current_time = time.time()
        if current_time - self.last_init_time < self.min_reinit_interval:
            return  # 避免过于频繁的重初始化
        with self.tracker_lock:
            try:
                # 获取当前帧进行重初始化
                frame_for_init = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame_for_init = self.latest_frame[0].copy()
                        index = self.latest_frame[1]
                if frame_for_init is not None:
                    bbox = self.reinit_func(frame_for_init)
                    if bbox:
                        # 重建追踪器
                        self.tracker = self._create_tracker()
                        self.tracker.init(frame_for_init, bbox)
                        self.current_bbox = bbox
                        # 计算中心点
                        center_x = int(bbox[0] + bbox[2] / 2)
                        center_y = int(bbox[1] + bbox[3] / 2)
                        self.tracking_position = [(center_x, center_y), 1]
                        self.last_init_time = current_time
                        logger.debug(f"Tracker reinitialized at frame {self.frame_count}")
            except Exception as e:
                logger.error(f"Reinitialization failed: {e}")

    def start_tracking(self, init_func: Callable[[np.ndarray], list]):
        """开始追踪"""
        self.reinit_func = init_func

        # 启动基础线程
        self.producer_thread = threading.Thread(
            target=self._frame_producer,
            name="FrameProducer",
            daemon=True
        )
        self.producer_thread.start()

        self.processor_thread1 = threading.Thread(
            target=self._tracking_processor,
            name="TrackingProcessor1",
            daemon=True
        )
        self.processor_thread2 = threading.Thread(
            target=self._tracking_processor,
            name="TrackingProcessor2",
            daemon=True
        )

        self.processor_thread1.start()
        self.processor_thread2.start()

        # 启动策略线程
        for thread in self.strategy_threads:
            thread.start()
        while True:
            logger.debug("real-time track running")
            time.sleep(60)

    def stop_tracking(self):
        """停止追踪"""
        self.running = False

        # 等待线程结束
        if hasattr(self, 'producer_thread'):
            self.producer_thread.join(timeout=1.0)
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join(timeout=1.0)

        # 清理资源
        if self.cap and self.cap.isOpened():
            self.cap.release()

        logger.info("Tracking stopped")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新的处理帧"""
        return self.latest_processed_frame.copy() if self.latest_processed_frame is not None else None

    def get_tracking_position(self) -> Optional[Tuple[int, int]]:
        """获取当前追踪位置"""
        return self.tracking_position

    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        return {
            'frame_count': self.frame_count,
            'process_count': self.process_count,
            'fps': self.process_count / max(time.time() - self.last_process_time, 0.001),
            'tracking_active': self.current_bbox is not None,
            'latency_ms': (time.time() - self.last_process_time) * 1000 if self.last_process_time > 0 else 0
        }


# 保持向后兼容性的包装类
class VideoStream(RealTimeTracker):
    def __init__(self, video_source, **kwargs):
        super().__init__(video_source, **kwargs)

    def track(self, func: Callable[[np.ndarray], list]):
        """兼容原有接口"""
        self.start_tracking(func)

    def next_track_frame(self):
        """兼容原有接口"""
        frame = self.get_latest_frame()
        return frame[0] if frame is not None else None

    def next_position(self):
        """兼容原有接口"""
        return self.get_tracking_position()

    def release(self):
        """兼容原有接口"""
        self.stop_tracking()
