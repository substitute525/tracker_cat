import cv2

from app import tracker

if __name__ == '__main__':
    stream = tracker.csrt.CsrtVideoStream(r"D:\Documents\WeChat Files\wxid_0ns0nrzywptg22\FileStorage\Video\2024-12\b00f8a9abdf70ebda25540c5fc180b45.mp4")
    ret, frame = stream.cap.read()
    bbox = tracker.select_box(frame)
    cv2.destroyAllWindows()
    stream.track(frame, bbox)
    while not stream.queue.empty():
        cv2.imshow("Tracking", stream.queue.get(False, 10))
        cv2.waitKey(1)
    stream.release()
