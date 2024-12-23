import typing as _typing

import cv2
from cv2.dnn import Net


class IModel:
    def load_model(self) -> Net:
        """
        模型加载

        :return: 模型
        """
        ...

    def forward(self, net: Net, layers: _typing.Sequence[str]) -> cv2.typing.MatLike:
        """
        运行模型推理

        :param net: 模型
        :param layers: 需要的层
        :return: 推理结果
        """
        ...

    def out_layers(self, net: Net, blob: cv2.UMat|cv2.typing.MatLike) -> list[str]:
        """
        获取模型推测的层

        :param net: 模型
        :param blob: 文件对象
        :return: layers
        """
        ...

    def class_names(self) -> list[str]:
        """
        类型名称

        :return:
        """
        ...

def box_layers(model: IModel, input: str):
    net = model.load_model()
    imread = cv2.imread(input)
    (h, w) = imread.shape[:2]
    blob = cv2.dnn.blobFromImage(imread, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    layers = model.out_layers(net, blob)
    box_layers(model, input, layers)


def box_layers(model: IModel, input: str, layers: list[str]):
    net = model.load_model()
    imread = cv2.imread(input)
    (h, w) = imread.shape[:2]

    # 推理
    model.forward(net, layers)
