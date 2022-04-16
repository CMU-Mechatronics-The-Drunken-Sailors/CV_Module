from pickle import FRAME
import cv2
import numpy as np
from scipy.signal import savgol_filter

from .common import threshold_for_color

FRAME_GROW_SIZE = 50


def getPose(pts):

    return center, angle


def get_stopcock_valve_position(frame, valve_bbox):
    # Crop to valve

    valve_frame = frame[
        max(0, int(valve_bbox[1]) - FRAME_GROW_SIZE) : min(
            frame.shape[0], int(valve_bbox[3]) + FRAME_GROW_SIZE
        ),
        max(0, int(valve_bbox[0]) - FRAME_GROW_SIZE) : min(
            frame.shape[1], int(valve_bbox[2]) + FRAME_GROW_SIZE
        ),
    ]

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
    frame_threshold = threshold_for_color(hsv_frame, color="blue_liberal")

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
    cv2.drawContours(frame_threshold, [hull], 0, 255, cv2.FILLED)

    # Use PCA Analysis to find center and orientation
    data_pts = np.empty((len(hull), 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = hull[i, 0, 0]
        data_pts[i, 1] = hull[i, 0, 1]

    # Perform PCA analysis
    _, eigenvectors, _ = cv2.PCACompute2(data_pts, np.empty((0)))

    # Find the center of the object
    M = cv2.moments(hull)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    valve_rotation = np.arctan2(
        eigenvectors[0, 1], eigenvectors[0, 0]
    )  # orientation in radians

    # We need to figure out which direction the center is. We know that the valve shape
    # has one rounded side, and one side with sharp corners. We find the rounded corners
    # side by fitting a rectangle to the shape, shrinking it slightly, and seeing which
    # corners do not intersect with the rectangle (those are the rounded corners). We then
    # flip the angle if necessary, and translate the center from the center of the shape to
    # the center of the actual valve.
    rect = cv2.minAreaRect(hull)
    # if rect[1][0] > rect[1][1]:
    #     rect = (rect[0], [r - min(rect[1]) * 0.1 for r in rect[1]], rect[2])

    # # Draw rectangle
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(frame_threshold, [box], 0, 150, 2)

    # Instead of seeing if a corner is inside the contour or not, we'll measure how far
    # each corner is from the contour, and choose the corner that is furthest away. That
    # corner corresponds to the side is the furthest away from the center of the valve.
    rect = (rect[0], rect[1], np.deg2rad(rect[2]))
    p1 = (
        rect[0][0]
        + 0.5 * (rect[1][0] * np.cos(rect[2]) - rect[1][1] * np.sin(rect[2])),
        rect[0][1]
        + 0.5 * (rect[1][0] * np.sin(rect[2]) + rect[1][1] * np.cos(rect[2])),
    )
    p2 = (
        rect[0][0]
        - 0.5 * (rect[1][0] * np.cos(rect[2]) - rect[1][1] * np.sin(rect[2])),
        rect[0][1]
        - 0.5 * (rect[1][0] * np.sin(rect[2]) + rect[1][1] * np.cos(rect[2])),
    )

    # # Draw points
    # cv2.circle(frame_threshold, np.int0(p1), 5, 120, -1)
    # cv2.circle(frame_threshold, np.int0(p2), 5, 120, -1)

    if cv2.pointPolygonTest(hull, p1, True) > cv2.pointPolygonTest(hull, p2, True):
        # Flip angle
        valve_rotation += np.pi
        if valve_rotation >= 360:
            valve_rotation -= 360
        elif valve_rotation < 0:
            valve_rotation += 360

    # Move center from center of shape to center of actual valve
    center = (
        int(center[0] - np.cos(valve_rotation) * 0.91 * rect[1][0]),
        int(center[1] - np.sin(valve_rotation) * 0.91 * rect[1][0]),
    )

    # # Draw line from center at angle rot
    # cv2.line(
    #     frame_threshold,
    #     center,
    #     (
    #         int(center[0] + np.cos(valve_rotation) * 300),
    #         int(center[1] + np.sin(valve_rotation) * 300),
    #     ),
    #     (0, 255, 0),
    #     2,
    # )
    # cv2.imshow("Valve", frame_threshold)

    return (
        valve_rotation,
        int(center[0] + valve_bbox[0] - FRAME_GROW_SIZE),
        int(center[1] + valve_bbox[1] - FRAME_GROW_SIZE),
    )
