import threading

import os
import sys

import args

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
    args.add_argument("video", type=str, help="Path to the input video file or Camera number")
    args.add_argument("classId", type=int, help="yolo model classification id. example: 15(cat)")
    args.add_argument("confidence", type=float, help="The model matches with minimum confidence. example: 0.1")

    args.add_argument("-d", "--debug", type=bool, default=False,
                        help="debug mode,will output the processed video stream. default: False")
    args.add_argument("-i", "--interval", type=int, default=0, help="frame extraction interval.default: 0")
    args.add_argument("--parallelism", type=int, default=1, help="calc threads num, Do not modify.default: 1")
    args.add_argument("-a", "--alg", type=lambda c: ALG[c.upper()], choices=list(ALG), default=ALG.KCF,
                        help="The tracking algorithm used.default: KCF")
    args.add_argument("--cache", type=int, default=100, help="calc queue size.default: 100")
    args.add_argument("-b", "--buffer", type=int, default=100, help="frame reade buffer.default: 100")

    args.add_argument("-s", "--strategy", action="append", type=lambda c: InitStrategy[c.upper()],
                        choices=list(InitStrategy), help="reInit strategies.example: InitStrategy.WHEN_LOST")
    args.add_argument("--minInterval", action="append", type=int,
                        help="reInit strategies, size must equal with strategy.example: 1000")
    args.add_argument("--maxInterval", action="append", type=int,
                        help="reInit strategies, size must equal with strategy.example: 0")
    args.add_argument("--strategyInterval", action="append", type=int,
                        help="reInit strategies, size must equal with strategy.example: 0")
    args.add_argument("--resize", nargs=2, type=int,
                        help="Scale the video to the target size(width height). example: 1920 1080")
    parser_args = args.get_args()
    print(parser_args)
    stream = track.tracker.VideoStream(int(parser_args.video) if parser_args.video.isdigit() else parser_args.video, save_frames = parser_args.debug,
                                       interval = parser_args.interval, parallelism = parser_args.parallelism, alg = parser_args.alg,
                                       calc_queue_size = parser_args.cache, frames_buffer_size = parser_args.buffer, resize = parser_args.resize)
    total_frame = int(stream.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = stream.cap.get(cv2.CAP_PROP_FPS)
    duration = 0 if fps == 0 else total_frame / fps
    start = time.time()
    yolo_ = yolo.yolo4.Yolo4()
    cv2.destroyAllWindows()
    strategies = list(parser_args.strategy)
    minInterval = list(parser_args.minInterval)
    maxInterval = list(parser_args.maxInterval)
    interval = list(parser_args.strategyInterval)
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

    def move():
        while True:
            _, position = stream.next_position()
            if position is None:
                print("无位置信息")
            else:
                print(position)
    threading.Thread(target=move).start()

    while True:
        _, frame = stream.next_track_frame()
        if frame is None and not track_thread.is_alive():
            break
        if frame is None:
            continue
        cv2.imshow("Tracking", frame)
        cv2.waitKey(max(1, int((1/fps) * 1000)))
    stream.release()
    cv2.destroyAllWindows()
