import cv2
import os
import numpy as np
import pickle
import enum


class BreakerState(enum.Enum):
    UP = 0
    DOWN = 1
    UP_UPSIDE_DOWN = 2
    DOWN_UPSIDE_DOWN = 3
    UNKNOWN = 4


BREAKER_STATE_CALIB = {
    BreakerState.DOWN_UPSIDE_DOWN: 0.79,
    BreakerState.UP_UPSIDE_DOWN: 0.61,
    BreakerState.UP: 0.24,
    BreakerState.DOWN: 0.43,
}

with open(os.path.join(os.path.dirname(__file__), "breaker_hsv.pkl"), "rb") as f:
    breaker_hsv = pickle.load(f)


def _threshold_for_color(hsv_img, color="breaker_switch"):
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

    morph_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, morph_ellipse)
    return frame_threshold


temp = []


def detect_breaker_state(frame, breaker_bbox):
    """Detects the breaker state of the breaker in the frame in the given bounding box.

    Args:
        frame (np.ndarray): The frame to detect the breaker state in. (opencv image)
        breaker_bbox (list(int)): The bounding box of the breaker in the frame (x1,y1,x2,y2)

    Returns:
        (BreakerState, float, float): (The state of the breaker, x value of the breaker switch, y value of the breaker switch)
    """
    # Crop to breaker
    breaker_frame = frame[
        int(breaker_bbox[1]) : int(breaker_bbox[3]),
        int(breaker_bbox[0]) : int(breaker_bbox[2]),
    ]

    aspect_ratio = breaker_frame.shape[1] / breaker_frame.shape[0]
    if aspect_ratio > 1.10:
        # Not a breaker
        return None

    area = (
        (breaker_bbox[2] - breaker_bbox[0])
        * (breaker_bbox[3] - breaker_bbox[1])
        / (frame.shape[0] * frame.shape[1])
    )  # Area as a fraction of the total frame size
    if area < 0.02:
        # Not a breaker
        return None

    # Convert to HSV
    hsv_frame = cv2.cvtColor(breaker_frame, cv2.COLOR_BGR2HSV)

    # Threshold for color
    frame_threshold = _threshold_for_color(hsv_frame)

    # Fill holes
    morph_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, morph_ellipse)

    # Find contours
    contours, _ = cv2.findContours(
        frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        # No contours
        return None

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find centroid
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Find normalized relative height of centroid with respect to the breaker
    switch_height = cy / breaker_frame.shape[0]

    # Identify breaker state by finding closest value in calib to switch_height
    breaker_state = min(
        BREAKER_STATE_CALIB.items(),
        key=lambda x: abs(x[1] - switch_height),
    )[0]

    return breaker_state, cx + int(breaker_bbox[0]), cy + int(breaker_bbox[1])

    # Draw centroid
    # cv2.circle(breaker_frame, (cx, cy), 5, (0, 0, 255), -1)
    # cv2.imshow("Breaker", breaker_frame)
