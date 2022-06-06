import time
import cv2
import mediapipe as mp
import numpy as np

from utils import FPS


class PoseDetector:
    def __init__(self, mode=False):
        self.mode = mode
        self.results = None
        self.img_shape = None

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode)  # , model_complexity=2)

    def findPose(self, img, draw=False):
        if self.img_shape is None:
            self.img_shape = img.shape

        img.flags.writeable = False
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
            img.flags.writeable = True
            self.mpDraw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS,
                # change color of drawing
                self.mpDraw.DrawingSpec(
                    color=(234, 115, 54), thickness=2, circle_radius=2
                ),  # in BGR not RGB
                self.mpDraw.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=2
                ),
            )
        return img

    def extract_coordinates_and_conf(self, landmark):
        """Extract x,y and z as array and confidence"""
        return np.array([landmark.x, landmark.y, landmark.z]), landmark.visibility

    def calculate_angle_3D(self, a, b, c):
        """ For calculating angle between two vectors in 3D space."""
        a, a_conf = self.extract_coordinates_and_conf(a)
        b, b_conf = self.extract_coordinates_and_conf(b)
        c, c_conf = self.extract_coordinates_and_conf(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return int(np.degrees(angle)), np.array([a_conf, b_conf, c_conf]).mean()

    def get_landmarks(self):
        if self.results and self.results.pose_landmarks:
            return self.results.pose_landmarks.landmark
        else:
            return []

    def is_skipping(self, lm):
        """ calculate angles between hand, hips and shoulders """
        left_angle = self.calculate_angle_3D(lm[15], lm[11], lm[23])[0]
        right_angle = self.calculate_angle_3D(lm[14], lm[12], lm[24])[0]
        avg_angle = (left_angle + right_angle) / 2

        if avg_angle > 22:
            return True
        else:
            return False

    def count_skipping(
        self, img, count, dir, is_skipping, baseline, position, draw=False
    ):
        """
        The skipping is counted by checking if the position of the feet of 
        the user cross a certain height threshold. This threshold is calculated by 
        considering the relativ height of the user.
        """
        h, w, c = self.img_shape
        lm = self.get_landmarks()
        avg_visibility = np.mean([m.visibility for m in lm])

        # jump heights
        distance_1_2 = 15
        distance_1_3 = 80

        # check if Person and Pose is detected
        anchor_relative = 0
        if len(lm) != 0:
            hips = (lm[23].y + lm[24].y) / 2
            shoulders = (lm[11].y + lm[12].y) / 2

            l_foot = lm[27].y
            r_foot = lm[28].y
            feet = l_foot if l_foot > r_foot else r_foot

            # get height of torso as factor: hips - shoulders
            height_Factor = hips - shoulders  # ~ 0.2 ... 0.4
            height_Factor = np.interp(height_Factor, (0.2, 0.4), (0.8, 1.2))

            distance_1_2 = distance_1_2 * height_Factor
            distance_1_3 = distance_1_3 * height_Factor

            anchor = feet

            if avg_visibility > 0.8 and self.is_skipping(lm):
                if not is_skipping:
                    baseline = anchor
                base = int(baseline * h)
                anchor_relative = int(anchor * h)
                is_skipping = True

                # Skipping-Counter mechanism
                line2 = base - distance_1_2
                line3 = base - distance_1_3

                if anchor_relative < line2:
                    if position == 0:  # first time over line 2
                        count += 1
                        position = 1
                if anchor_relative > line2:
                    if position == 1:
                        position = 0
                if anchor_relative < line3:
                    if position == 1:  # first time over line 3
                        count += 1
                        position = 2
                if anchor_relative > line3:  # double jump
                    if position == 2:
                        position = 1

            else:
                is_skipping = False
                base = int(anchor * h)
        else:
            base = 0
            hips = 0
            shoulders = 0

        if draw:
            cv2.line(img, (0, base - 10), (w, base), (0, 0, 255), 3)
            first = int(base - distance_1_2)
            cv2.line(img, (0, first), (w, first), (0, 0, 255), 3)
            second = int(base - distance_1_3)
            cv2.line(img, (0, second), (w, second), (0, 0, 255), 3)
            cv2.line(img, (0, anchor_relative), (w, anchor_relative), (0, 255, 0), 3)

        return img, count, dir, is_skipping, baseline, position

    def skipping_speed(self, history, time, count):
        buffer = 5  # sec
        history = np.roll(history, -2)
        history[-1] = np.array([time, count])
        skipps_since = 0
        for i in history:
            if i[0] >= (time - buffer):
                skipps_since = count - i[1]
                break

        speed = skipps_since * (60 / buffer)  # per minute

        return history, speed


class SkipCounter:
    def __init__(self):
        self.detector = PoseDetector()
        self.vidcap = cv2.VideoCapture(
            # 0
            "videos/skipping_standard.mp4"
            # "videos/skipping_doubleunder.mp4"
            # "videos/skipping_sidetoside.mp4"
            # "videos/skipping_standard+sidetoside.mp4"
            # "videos/skipping_test_full.mp4",
        )  # , cv2.CAP_V4L2)
        self.init_specs()

    def init_specs(self):
        self.fps = FPS().start()
        self.start = time.time()
        self.previous_time = time.time()
        self.skipping_time = 0
        self.break_time = 0
        self.count = 0
        self.dir = 0
        self.is_skipping = False
        self.position = 0
        self.baseline = 0.5
        self.history_list = np.zeros((50, 2))
        self.speed = 0
        self.i = 0
        self.speed_list = []

    def return_as_byte(self, img):
        ret, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()

    def get_frame(self):
        success, frame = self.vidcap.read()

        # loop video
        if not success:
            self.vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.vidcap.read()

        frame = self.detector.findPose(frame, draw=True)
        (
            frame,
            self.count,
            self.dir,
            self.is_skipping,
            self.baseline,
            self.position,
        ) = self.detector.count_skipping(
            frame, self.count, self.dir, self.is_skipping, self.baseline, self.position
        )
        self.fps.update()
        return self.return_as_byte(frame)

    def get_data(self):
        if self.break_time is not 0:
            break_delta = time.time() - self.break_time
            self.start = self.start - break_delta
            self.skipping_time = self.skipping_time - break_delta
            self.break_time = 0

        # times
        total_time = time.time() - self.start
        if self.is_skipping:
            self.skipping_time += time.time() - self.previous_time
        break_time = total_time - self.skipping_time

        # speed
        if self.i == 2:
            self.history_list, self.speed = self.detector.skipping_speed(
                self.history_list, total_time, self.count
            )
            self.i = 0
        self.speed_list.append([total_time, self.speed])
        max_speed = np.asarray(self.speed_list)[:, 1].max()

        fps_str = "FPS: {:.2f}".format(self.fps.fps())

        self.previous_time = time.time()
        self.i += 1
        return (
            self.count,
            total_time,
            self.skipping_time,
            break_time,
            self.speed,
            np.asarray(self.speed_list),
            max_speed,
            self.is_skipping,
            fps_str,
        )
