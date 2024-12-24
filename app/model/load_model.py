import typing

import cv2
import numpy as np


# 加载模型
def load_model():
    weights_path = r"./yolov4.weights"
    cfg_path = r"./yolov4.cfg"
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def forward(net: Net, layers: typing.Sequence[str]):


net = load_model()
imread = cv2.imread(r"D:\Pictures\1202017149.jpg")
(h, w) = imread.shape[:2]

blob = cv2.dnn.blobFromImage(imread, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

outputs = net.forward(output_layers)

# 处理检测结果
boxes = []
confidences = []
class_ids = []
class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
               'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 非极大值抑制（NMS）过滤重叠框
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in indices:
    box = boxes[i]
    class_id = class_ids[i]
    (x, y) = (box[0], box[1])
    (w1, h1) = (box[2], box[3])
    x = max(0, x)  # 防止框的左上角超出左边界
    y = max(0, y)  # 防止框的左上角超出上边界
    x2 = min(w, x + w1)  # 防止框的右下角超出右边界
    y2 = min(h, y + h1)  # 防止框的右下角超出下边界
    cv2.rectangle(imread, (x, y), (x2, y2), (0, 255, 0), 2)
    # 添加标签文本
    cv2.putText(imread, class_names[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)
# 显示结果
w_ = 1080 / w
h_ = 1920 / h
f = min(w_, h_)

imread = cv2.resize(imread, (int(w * f), int(h * f)))
cv2.imshow("Image", imread)
cv2.waitKey(0)
cv2.destroyAllWindows()
