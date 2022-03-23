import cv2
import os
import yolov5

WEIGHTS = os.path.join(os.path.dirname(__file__), "model.pt")
model = yolov5.load(WEIGHTS)

def apply_nn(frame):
    results = model(frame)
    print(results)
    
    return results