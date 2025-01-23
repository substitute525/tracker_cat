import threading
import argparse

import os
import sys
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import config
config.init_config()

import time

import cv2

from app import track
from app.model import yolo
from app.model import yolo_most_like_box
from app.track.tracker import InitStrategy, ALG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument("video", type=str, help="Path to the input video file or Camera number")
    parser.add_argument("classId", type=int, help="yolo model classification id. example: 15(cat)")
    parser.add_argument("confidence", type=float, help="The model matches with minimum confidence. example: 0.1")

    parser.add_argument("-d", "--debug", type=bool, default=False, help="debug mode,will output the processed video stream. default: False")
    parser.add_argument("-i", "--interval", type=int, default=0, help="frame extraction interval.default: 0")
    parser.add_argument("--parallelism", type=int, default=1, help="calc threads num, Do not modify.default: 1")
    parser.add_argument("-a","--alg", type=lambda c: ALG[c.upper()], choices=list(ALG), default=ALG.KCF, help="The tracking algorithm used.default: KCF")
    parser.add_argument("--cache", type=int, default=100, help="calc queue size.default: 100")
    parser.add_argument("-b","--buffer", type=int, default=100, help="frame reade buffer.default: 100")

    parser.add_argument("-s","--strategy", action="append", type=lambda c: InitStrategy[c.upper()], choices=list(InitStrategy), help="reInit strategies.example: InitStrategy.WHEN_LOST")
    parser.add_argument("--minInterval", action="append", type=int, help="reInit strategies, size must equal with strategy.example: 1000")
    parser.add_argument("--maxInterval", action="append", type=int, help="reInit strategies, size must equal with strategy.example: 0")
    parser.add_argument("--strategyInterval", action="append", type=int, help="reInit strategies, size must equal with strategy.example: 0")

    args = parser.parse_args()
    print(args)
    stream = track.tracker.VideoStream(args.video, save_frames = args.debug, interval = args.interval, parallelism = args.parallelism, alg = args.alg, calc_queue_size = args.cache, frames_buffer_size = args.buffer)
    total_frame = int(stream.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = stream.cap.get(cv2.CAP_PROP_FPS)
    duration = total_frame / fps
    start = time.time()
    yolo_ = yolo.yolo4.Yolo4()
    cv2.destroyAllWindows()
    strategies = list(args.strategy)
    minInterval = list(args.minInterval)
    maxInterval = list(args.maxInterval)
    interval = list(args.strategyInterval)
    for i, strategy in enumerate(strategies):
        stream.re_init_strategy(strategy=strategy, min_interval=minInterval[i], max_interval=maxInterval[i], interval=interval[i])
    track_thread = threading.Thread(
        target=stream.track,
        args=(lambda frm: yolo_most_like_box(yolo_, frm, confidence=0.1, class_id=15)[1],)
    )
    track_thread.start()
    def log():
        track_thread.join()
        print(f'track cost:{time.time() - start}, total frame:{total_frame}, fps:{fps}, video duration:{duration}')
    threading.Thread(target=log).start()

    while True:
        _, frame = stream.next_track_frame()
        if frame is None and not track_thread.is_alive():
            break
        if frame is None:
            continue
        cv2.imshow("Tracking", frame)
        cv2.waitKey(10)
    stream.release()
    cv2.destroyAllWindows()
