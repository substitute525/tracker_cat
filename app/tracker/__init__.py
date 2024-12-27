import cv2

from . import csrt

def select_box(frame):
    return cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
