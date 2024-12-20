import cv2
import numpy as np

# 加载模型
onnx_path = r"D:/self/tracker_cat/app/yolo/tiny-yolov3-11.onnx"
weights_path = r"D:\self\tracker_cat\app\yolo\yolov3-tiny.weights"
cfg_path = r"D:\self\tracker_cat\app\yolo\yolov3-tiny.cfg"
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

imread = cv2.imread("D:\\1202017149.jpg")
(h, w) = imread.shape[:2]

blob = cv2.dnn.blobFromImage(imread, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

outputs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

for detection in outputs[0, 0]:  # 按照 YOLO 输出格式解析
    scores = detection[5:]  # 提取类别概率
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    if confidence > 0.5:  # 设置置信度阈值
        box = detection[:4] * np.array([w, h, w, h])
        (centerX, centerY, width, height) = box.astype("int")
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        class_ids.append(class_id)

# 绘制检测框
for i, box in enumerate(boxes):
    (x, y, width, height) = box
    cv2.rectangle(imread, (x, y), (x + width, y + height), (0, 255, 0), 2)
    text = f"Class {class_ids[i]}: {confidences[i]:.2f}"
    cv2.putText(imread, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果
cv2.imshow("YOLO Detection", imread)
cv2.waitKey(0)
cv2.destroyAllWindows()
