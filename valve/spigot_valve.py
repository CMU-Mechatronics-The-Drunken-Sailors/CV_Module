from pickle import FRAME
import cv2
import numpy as np
from scipy.signal import savgol_filter

from .common import threshold_for_color

NUM_AVERAGE_POINTS = 10
FRAME_GROW_SIZE = 50

def _moving_average(a, n=NUM_AVERAGE_POINTS) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_spigot_valve_position(frame, valve_bbox):
    # Crop to valve

    valve_frame = frame[
        max(0, int(valve_bbox[1]) - FRAME_GROW_SIZE) : min(frame.shape[0], int(valve_bbox[3]) + FRAME_GROW_SIZE),
        max(0, int(valve_bbox[0]) - FRAME_GROW_SIZE) : min(frame.shape[1], int(valve_bbox[2]) + FRAME_GROW_SIZE),
    ]

    aspect_ratio = valve_frame.shape[1] / valve_frame.shape[0]
    if abs(aspect_ratio - 1) > 0.1:
        # Not a valve
        return None

    area = (
        (valve_bbox[2] - valve_bbox[0])
        * (valve_bbox[3] - valve_bbox[1])
        / (frame.shape[0] * frame.shape[1])
    )  # Area as a fraction of the total frame size
    if area < 0.005:
        # Not a valve
        return None

    # Convert to HSV
    blurred_frame = cv2.GaussianBlur(valve_frame, (11, 11), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Threshold for color
    frame_threshold = threshold_for_color(hsv_frame, color="blue")

    # Find contours
    contours, _ = cv2.findContours(
        frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        # No contours
        return None

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Convex hull of largest contour, and enclosing circle
    hull = cv2.convexHull(largest_contour)
    outerRing = list(cv2.minEnclosingCircle(hull))

    # Find angle
    outerRing[1] *= 0.675

    frame_threshold = threshold_for_color(hsv_frame, color="blue_liberal")

    # Fill all the holes by finding the valve outline and drawing that shape
    contours, _ = cv2.findContours(
        frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(frame_threshold, [largest_contour], 0, 255, cv2.FILLED)

    # Find the divot in the valve outline (which is the tape marking)
    thresh_along_circle = [
        np.sum(
            frame_threshold[
                int(outerRing[0][1] + outerRing[1] * np.sin(angle)),
                int(outerRing[0][0] + outerRing[1] * np.cos(angle))
            ],
        )
        for angle in np.arange(360 + NUM_AVERAGE_POINTS) * np.pi / 180
    ]

    # Smooth data using savitsky-golay filter, then average 10-largest values
    thresh_along_circle = savgol_filter(thresh_along_circle, 11, 3)
    moving_avg = _moving_average(thresh_along_circle)
    valve_rotation = np.argmin(moving_avg) + NUM_AVERAGE_POINTS / 2

    # Check for wraparound case
    if valve_rotation >= 360:
        valve_rotation -= 360
    elif valve_rotation < 0:
        valve_rotation += 360

    valve_rotation = np.deg2rad(valve_rotation)

    # # Draw line from center at angle rot
    # cv2.line(
    #     frame_threshold,
    #     [int(x) for x in outerRing[0]],
    #     (
    #         int(outerRing[0][0] + np.cos(valve_rotation) * 100),
    #         int(outerRing[0][1] + np.sin(valve_rotation) * 100),
    #     ),
    #     (0, 255, 0),
    #     2,
    # )
    # cv2.circle(
    #     frame_threshold,
    #     (int(outerRing[0][0]), int(outerRing[0][1])),
    #     int(outerRing[1]),
    #     (0, 255, 0),
    #     2,
    # )
    # cv2.circle(
    #     frame_threshold, (int(outerRing[0][0]), int(outerRing[0][1])), 2, (0, 255, 0), 3
    # )
    # cv2.imshow("Valve", frame_threshold)

    return (
        valve_rotation,
        int(outerRing[0][0] + valve_bbox[0] - FRAME_GROW_SIZE),
        int(outerRing[0][1] + valve_bbox[1] - FRAME_GROW_SIZE),
    )
