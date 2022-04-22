from enum import Enum
import cv2
import os
# import torch
import yolov5

WEIGHTS = os.path.join(os.path.dirname(__file__), "model.pt")
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=WEIGHTS)
model = yolov5.load(WEIGHTS)


class NN_Labels(Enum):
    BREAKER = 0
    SPIGOTTOPVIEW = 1
    SPIGOTSIDEVIEW = 2
    STOPCOCKTOPVIEW = 3
    STOPCOCKSIDEVIEW = 4
    ROTARY = 5

    def __int__(self):
        return self.value

    def __eq__(self, other):
        if(type(other) == int):
            return self.value == other
        else:
            return super().__eq__(other)


def apply_nn(frame):
    results = model(frame)
    # print(results.xyxy)

    return results
