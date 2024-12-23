import cv2

from . import csrt

def selectBox(frame):
    return cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)