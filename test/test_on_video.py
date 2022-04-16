import cv2
import numpy as np
import os

from nn import apply_nn, NN_Labels
from breaker import detect_breaker_state, BreakerState
from valve import get_rotary_valve_position, get_spigot_valve_position, get_stopcock_valve_position


def filter_video(video_filename):
    """
    Modify this as needed to filter out certian video files
    """
    return "" in video_filename


if __name__ == "__main__":
    # Load validation videos
    video_list = []

    root = os.path.join(os.path.dirname(__file__), "data")

    for filename in os.listdir(root):
        file_path = os.path.join(root, filename)
        file_name, file_ext = os.path.splitext(file_path)

        # If the file is an video...
        if (file_ext == ".MOV" or file_ext == ".mp4") and filter_video(file_name):
            # Add to list
            video_list.append((file_name, file_ext))

    # For each video...
    for video_path, video_ext in video_list:
        video = cv2.VideoCapture(video_path + video_ext)

        # For each frame...
        while True:
            for _ in range(30):
                video.grab()

            # Read frame
            ret, frame = video.read()

            # If the frame is empty...
            if frame is None:
                break

            # Resize frame to 1280x720
            frame = cv2.resize(frame, (1280, 720))

            results = apply_nn(frame)
            for x1, y1, x2, y2, _, class_id in results.xyxy[0]:
                # Convert from pytorch tensor to int
                class_id = int(class_id)
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                # Check that width and height are greater than zero
                if x2 - x1 > 0 and y2 - y1 > 0:

                    if class_id == NN_Labels.BREAKER:
                        res = detect_breaker_state(frame, (x1, y1, x2, y2))
                        if res is not None:
                            breaker_state, bx, by = res

                            if breaker_state == BreakerState.UP:
                                color = (255, 0, 0)
                            elif breaker_state == BreakerState.DOWN:
                                color = (0, 255, 0)
                            elif breaker_state == BreakerState.UP_UPSIDE_DOWN:
                                color = (0, 0, 255)
                            elif breaker_state == BreakerState.DOWN_UPSIDE_DOWN:
                                color = (255, 255, 0)

                            cv2.circle(frame, (bx, by), 10, color, -1)

                    elif class_id == NN_Labels.ROTARY:
                        res = get_rotary_valve_position(frame, (x1, y1, x2, y2))
                        if res is not None:
                            rot, bx, by = res

                            # Draw line from center at angle rot
                            cv2.line(
                                frame,
                                (bx, by),
                                (bx + int(np.cos(rot) * 100), by + int(np.sin(rot) * 100)),
                                (0, 255, 0),
                                2,
                            )

                    elif class_id == NN_Labels.SPIGOTSIDEVIEW:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    elif class_id == NN_Labels.SPIGOTTOPVIEW:
                        res = get_spigot_valve_position(frame, (x1, y1, x2, y2))
                        if res is not None:
                            rot, bx, by = res

                            # Draw line from center at angle rot
                            cv2.line(
                                frame,
                                (bx, by),
                                (bx + int(np.cos(rot) * 100), by + int(np.sin(rot) * 100)),
                                (0, 255, 0),
                                2,
                            )

                    elif class_id == NN_Labels.STOPCOCKSIDEVIEW:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    elif class_id == NN_Labels.STOPCOCKTOPVIEW:
                        res = get_stopcock_valve_position(frame, (x1, y1, x2, y2))
                        if res is not None:
                            rot, bx, by = res

                            # Draw line from center at angle rot
                            cv2.line(
                                frame,
                                (bx, by),
                                (bx + int(np.cos(rot) * 300), by + int(np.sin(rot) * 300)),
                                (0, 255, 0),
                                2,
                            )

            # Display YOLO Results frame
            imgs = results.render()
            cv2.imshow("Frame", imgs[0])

            # If the user presses the 'q' key...
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
