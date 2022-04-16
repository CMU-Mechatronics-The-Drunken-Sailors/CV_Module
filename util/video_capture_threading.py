import cv2
from datetime import datetime
import numpy as np
import pickle
import threading

#################################################################################################################
#################################################################################################################       
#### Source: https://github.com/gilbertfrancois/video-capture-async/blob/master/main/gfd/py/video/capture.py ####
#################################################################################################################
#################################################################################################################

class VideoCapture:
    def __init__(self, port=0, width=3840, height=2160, fps=30, calib=""):
        self.port = port
        self.cap = cv2.VideoCapture(self.port)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if calib is not None and calib != "":
            with open(calib, "rb") as f:
                self.calib = pickle.load(f)
        
        # self.writer = cv2.VideoWriter("videos/out_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.avi",
        #                               cv2.VideoWriter_fourcc("*MJPG"), 30, width, height)

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        return self

    def read(self):
        frames_per_loop = int(30 / self.cap.get(cv2.CAP_PROP_FPS))
        for i in range(frames_per_loop):
            self.cap.grab()
        grabbed, frame = self.cap.retrieve()
        # self.writer.write(frame)

        # Perform white balancing
        brightest_coords = np.argmax(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        brightest_pix = frame.reshape((-1, 3))[brightest_coords]
        frame[:,:,0] = np.array(np.array(frame[:,:,0], dtype=np.float32) * 255.0 / brightest_pix[0], dtype=np.uint8)
        frame[:,:,1] = np.array(np.array(frame[:,:,1], dtype=np.float32) * 255.0 / brightest_pix[1], dtype=np.uint8)
        frame[:,:,2] = np.array(np.array(frame[:,:,2], dtype=np.float32) * 255.0 / brightest_pix[2], dtype=np.uint8)

        return grabbed, frame

    def read_calib(self):
        grabbed, frame = self.read()
        frame = cv2.undistort(frame, self.calib["camera_matrix"], self.calib["distortion"])
        return grabbed, frame
    
    def stop(self):
        return

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
        # self.writer.release()


class VideoCaptureThreading:
    def __init__(self, port=0, width=3840, height=2160, fps=30, calib=""):
        self.port = port
        self.cap = cv2.VideoCapture(self.port)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

        if calib is not None and calib != "":
            with open(calib, "rb") as f:
                self.calib = pickle.load(f)
        
        # self.writer = cv2.VideoWriter(f"videos/out_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.mp4",
        #                               cv2.VideoWriter_fourcc(*"H264"), 30, (width, height))

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def grab(self):
        self.cap.grab()

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            # self.writer.write(frame)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        
        # Perform white balancing
        brightest_coords = np.argmax(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        brightest_pix = frame.reshape((-1, 3))[brightest_coords]
        frame[:,:,0] = np.array(np.array(frame[:,:,0], dtype=np.float32) * 255.0 / brightest_pix[0], dtype=np.uint8)
        frame[:,:,1] = np.array(np.array(frame[:,:,1], dtype=np.float32) * 255.0 / brightest_pix[1], dtype=np.uint8)
        frame[:,:,2] = np.array(np.array(frame[:,:,2], dtype=np.float32) * 255.0 / brightest_pix[2], dtype=np.uint8)

        return grabbed, frame

    def read_calib(self):
        grabbed, frame = self.read()
        frame = cv2.undistort(frame, self.calib["camera_matrix"], self.calib["distortion"])
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()
    
    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
        # self.writer.release()
