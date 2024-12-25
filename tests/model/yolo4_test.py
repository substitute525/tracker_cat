import cv2

from app import model
from app.model.yolo import yolo4

if __name__ == '__main__':
    yolo_ = yolo4.Yolo4()
    imread, box = model.yolo_most_like_box(yolo_, r"D:\IMG_20240831_234839.jpg", 15, 0.1)
    if not box:
        print("未找到")
        exit()
    h, w = imread.shape[:2]
    (x, y) = (box[0], box[1])
    (w1, h1) = (box[2], box[3])
    x = max(0, x)  # 防止框的左上角超出左边界
    y = max(0, y)  # 防止框的左上角超出上边界
    x2 = min(w, x + w1)  # 防止框的右下角超出右边界
    y2 = min(h, y + h1)  # 防止框的右下角超出下边界
    cv2.rectangle(imread, (x, y), (x2, y2), (0, 255, 0), 2)
    # 显示结果
    w_ = 1920 / w
    h_ = 1080 / h
    f = min(w_, h_)
    imread = cv2.resize(imread, (int(w * f), int(h * f)))
    cv2.imshow("Image", imread)
    cv2.waitKey()
    cv2.destroyAllWindows()

