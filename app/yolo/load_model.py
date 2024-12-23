import cv2
import numpy as np

# 加载模型
onnx_path = r"D:/self/tracker_cat/app/yolo/tiny-yolov3-11.onnx"
weights_path = r"D:\self\tracker_cat\app\yolo\yolov3-tiny.weights"
cfg_path = r"D:\self\tracker_cat\app\yolo\yolov3-tiny.cfg"
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

imread = cv2.imread("D:\\1202017149.jpg")
(h, w) = imread.shape[:2]

blob = cv2.dnn.blobFromImage(imread, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

outputs = net.forward(output_layers[0])

# 处理检测结果
boxes = []
confidences = []
class_ids = []
for output in outputs:
    for detection in output:
        scores = detection
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
    i = i[0]
    box = boxes[i]
    (x, y) = (box[0], box[1])
    (w, h) = (box[2], box[3])
    cv2.rectangle(imread, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
cv2.imshow("Image", imread)
cv2.waitKey(0)
cv2.destroyAllWindows()
