import os
import pickle
import cv2

with open(os.path.join(os.path.dirname(__file__), "valve_hsv.pkl"), "rb") as f:
    breaker_hsv = pickle.load(f)


def threshold_for_color(hsv_img, color="blue"):
    """
    Options for color: "blue", "blue_liberal"
    """
    [color_min_hsv, color_max_hsv] = breaker_hsv

    (low_H, low_S, low_V) = color_min_hsv[color]
    (high_H, high_S, high_V) = color_max_hsv[color]
    if low_H > high_H:
        # Wrap around
        frame_threshold = cv2.bitwise_or(
            cv2.inRange(hsv_img, (low_H, low_S, low_V), (360, high_S, high_V)),
            cv2.inRange(hsv_img, (0, low_S, low_V), (high_H, high_S, high_V)),
        )
    else:
        frame_threshold = cv2.inRange(
            hsv_img, (low_H, low_S, low_V), (high_H, high_S, high_V)
        )

    morph_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, morph_ellipse)
    return frame_threshold
