import cv2


def track(video: int|str):
    cap = cv2.VideoCapture(video)
    tracker = cv2.TrackerCSRT_create()
    _, frame = cap.read()
    bbox = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost Tracking", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Tracking", frame)
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def release(cap: cv2.VideoCapture):
    cap.release()
    cv2.destroyAllWindows()