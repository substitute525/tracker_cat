import typing as _typing

import cv2

from app.model import IModel


class Yolo4(IModel):
    def load_model(self):
        weights_path = r"./yolov4.weights"
        cfg_path = r"./yolov4.cfg"
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def forward(self, net, layers: _typing.Sequence[str]):
        imread = cv2.imread(r"D:\Pictures\1202017149.jpg")
        (h, w) = imread.shape[:2]

        blob = cv2.dnn.blobFromImage(imread, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)


        return net.forward(layers)

    def out_layers(self, net, blob) -> list[str]:
        return super().out_layers(net, blob)