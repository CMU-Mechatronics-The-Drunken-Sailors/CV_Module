import argparse
import cv2
import numpy as np
import pickle
import pyrealsense2 as rs

from video_capture_threading import VideoCaptureThreading as VideoCapture

KEYS = ["blue", "blue_liberal", "stopcock_blue"]
FILE = "valve/valve_hsv.pkl"

# KEYS = ["breaker_switch"]
# FILE = "breaker/breaker_hsv.pkl"

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_detection_name = "Object Detection"
low_H_name = "Low H"
low_S_name = "Low S"
low_V_name = "Low V"
high_H_name = "High H"
high_S_name = "High S"
high_V_name = "High V"


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    # low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    # high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)


if __name__ == "__main__":
    # Command line argument parsing
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--cam_port", "-p", type=str, default=0, help="OpenCV camera port or video file"
    # )
    # parser.add_argument(
    #     "--cap_width", "-x", type=int, default=3840, help="Camera capture width"
    # )
    # parser.add_argument(
    #     "--cap_height", "-y", type=int, default=2160, help="Camera capture height"
    # )
    # parser.add_argument(
    #     "--cap_fps", "-f", type=int, default=30, help="Camera capture FPS"
    # )
    # parser.add_argument(
    #     "--cam_calib",
    #     "-c",
    #     type=str,
    #     default="camera_calibration_data.pkl",
    #     help="Camera calibration",
    # )
    # parser.add_argument("--use_calib", "-u", action="store_true")
    # args = parser.parse_args()

    # # Read frames from webcam
    # if args.cam_port.isdigit():
    #     cap = VideoCapture(
    #         port=int(args.cam_port),
    #         width=args.cap_width,
    #         height=args.cap_height,
    #         fps=args.cap_fps,
    #         calib=args.cam_calib,
    #     ).start()
    # else:
    #     cap = cv2.VideoCapture(args.cam_port)
    pipeline = rs.pipeline()
    config = rs.config()

    # get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    # Check if the camera has RGB color channels set up
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(1)

    # enable depth image streaming
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

    # enable color image streaming
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 15)
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

    device.sensors[1].set_option(rs.option.white_balance, 3600) # 2800-6500/10 [4600]
    device.sensors[1].set_option(rs.option.saturation, 66) # 0-100/1 [64]
    device.sensors[1].set_option(rs.option.hue, 25) # -180-180/1 [0]
    device.sensors[1].set_option(rs.option.exposure, 200) # 1-10000/1 [166]
    print(device.sensors[1].get_option(rs.option.exposure))
    print(device.sensors[1].get_option_range(rs.option.exposure))

    # start streaming
    pipeline.start(config)



    # Create calibration window
    cv2.namedWindow(window_detection_name)
    cv2.createTrackbar(
        low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar
    )
    cv2.createTrackbar(
        high_H_name,
        window_detection_name,
        high_H,
        max_value_H,
        on_high_H_thresh_trackbar,
    )
    cv2.createTrackbar(
        low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar
    )
    cv2.createTrackbar(
        high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar
    )
    cv2.createTrackbar(
        low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar
    )
    cv2.createTrackbar(
        high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar
    )

    # Collect ROIs for all the block colors
    try:
        with open(FILE, "rb") as f:
            [color_min_hsv, color_max_hsv] = pickle.load(f)
            print("Loaded HSV calibration data: ", color_min_hsv, color_max_hsv)
            print(color_min_hsv)
            print(color_max_hsv)
    except:
        print("Could not find calibration data. Creating a new one...")
        color_min_hsv = {}
        color_max_hsv = {}


    for col in KEYS:
        if col in color_min_hsv and col in color_max_hsv:
            (low_H, low_S, low_V) = color_min_hsv[col]
            (high_H, high_S, high_V) = color_max_hsv[col]
        cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
        cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
        cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
        cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
        cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
        cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

        while True:
            # Read frame (skip a few frames so we can seek the video faster)
            frames = pipeline.wait_for_frames()
            frame = np.asanyarray(frames.get_color_frame().get_data())

            if frame is None:
                break

            # Resize frame
            frame = cv2.resize(frame, (640, 360))

            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            if low_H > high_H:
                # Wrap around
                frame_threshold = cv2.bitwise_or(
                    cv2.inRange(
                        frame_HSV, (low_H, low_S, low_V), (max_value_H, high_S, high_V)
                    ),
                    cv2.inRange(frame_HSV, (0, low_S, low_V), (high_H, high_S, high_V)),
                )
            else:
                frame_threshold = cv2.inRange(
                    frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V)
                )

            morph_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, morph_ellipse)

            frame_combined = cv2.max(
                frame, np.repeat(frame_threshold[:, :, np.newaxis], 3, axis=2)
            )
            frame_with_text = cv2.putText(
                frame_combined,
                f"Adjust sliders to select only {col}:",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (100, 100, 100),
                1,
            )

            cv2.imshow(window_detection_name, frame_with_text)

            key = cv2.waitKey(30)
            if key == ord("q") or key == 27:
                break

        color_min_hsv[col] = [low_H, low_S, low_V]
        color_max_hsv[col] = [high_H, high_S, high_V]

    # Save the mean and standard deviation values to a pickle file
    with open(FILE, "wb") as f:
        pickle.dump([color_min_hsv, color_max_hsv], f)

    # Stop the camera
    cv2.destroyAllWindows()
