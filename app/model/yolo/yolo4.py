import typing as _typing

import cv2
import numpy as np

from app.model import IModel


class Yolo4(IModel):
    def __init__(self):
        weights_path = r"./yolov4.weights"
        cfg_path = r"./yolov4.cfg"
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def reset_blob(self, image_path):
        imread = cv2.imread(image_path)
        blob = cv2.dnn.blobFromImage(imread, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        return imread

    def get_out_layers(self) -> _typing.Sequence[str]:
        layer_names = self.net.getLayerNames()
        return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def class_names(self) -> list[str]:
        with open('filename.txt', 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
        return lines

    def forward(self, layers: _typing.Sequence[str]):
        return self.net.forward(layers)

    def output_process(self, outputs, confidence: float = 0.5, **kwargs) -> tuple[
        list[list[int]], list[float], list[int]]:
        w = kwargs.get('w')
        h = kwargs.get('h')
        # 输出结果处理，保留高置信度
        boxes = []
        confidences = []
        class_ids = []
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
        return boxes, confidences, class_ids

