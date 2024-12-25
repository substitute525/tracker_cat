import typing as _typing

import cv2
import numpy
import numpy as np
from numpy import ndarray


class IModel:
    def reset_blob(self, file) -> _typing.Any:
        """
        重置模型解析的文件

        :param file: 文件或文件路径
        :return: blob
        """
        ...

    def get_out_layers(self) -> _typing.Sequence[str]:
        """
        模型推测可能的层

        :return: 层数组
        """

    def forward(self, layers: _typing.Sequence[str]) -> _typing.Any:
        """
        模型推理

        :param layers: 需要推理的层
        :return: 推理结果
        """
        ...

    def class_names(self) -> list[str]:
        """
        类型名称

        :return:
        """
        ...

    def output_process(self, outputs, confidence: float = 0.5, **kwargs) -> tuple[
        list[list[int]], list[float], list[int]]:
        """
        模型输出结果后置

        :param outputs:
        :param confidence:
        :return: (边框，置信度，类别)
        """

def yolo_most_like_box(model: IModel, file, class_id:int=None, confidence:float=0.5) -> (ndarray, list[int]):
    """
    yolo模型-最相似框
    :param model: 模型对象
    :param file: 文件（path/ndarray）
    :param class_id: 类型id
    :param confidence: 最低置信度
    :return: (文件, 范围框)
    """
    imread:ndarray = model.reset_blob(file)
    h, w = imread.shape[:2]
    layers = model.get_out_layers()
    print(layers)
    outputs = model.forward(layers)
    class_names = model.class_names()
    boxes, confidences, class_ids = model.output_process(outputs, confidence=confidence, w=w, h=h, class_id=class_id)
    if not class_ids:
        return imread, None
    index = np.argmax(numpy.array(confidences))
    class_id = class_names[class_ids[index]]
    print(confidences[index], class_id)
    box = boxes[index]
    return imread, box



def example():
    from .yolo.yolo4 import Yolo4
    _yolo = Yolo4()
    imread = _yolo.reset_blob(r"D:\IMG_20240831_234839.jpg")
    h, w = imread.shape[:2]
    layers = _yolo.get_out_layers()
    print(layers)
    outputs = _yolo.forward(layers)
    class_names = _yolo.class_names()
    boxes, confidences, class_ids = _yolo.output_process(outputs, confidence=0.1, w=w, h=h, class_id=15)

    # 非极大值抑制（NMS）过滤重叠框
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)
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
    cv2.waitKey()
    cv2.destroyAllWindows()
