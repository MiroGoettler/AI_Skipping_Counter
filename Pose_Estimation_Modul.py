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
        self.pose = self.mpPose.Pose(self.mode)

    def find_pose(self, img, draw=False):
        if self.img_shape is None:
            self.img_shape = img.shape

        img.flags.writeable = False
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # convert img to grey
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        landmark_color = (255, 0, 0)
        if draw and self.results.pose_landmarks:
            img.flags.writeable = True
            self.mpDraw.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mpPose.POSE_CONNECTIONS,
                # change color of drawing
                self.mpDraw.DrawingSpec(
                    color=landmark_color, thickness=2, circle_radius=2
                ),  # in BGR NOT RGB
                self.mpDraw.DrawingSpec(
                    color=landmark_color, thickness=2, circle_radius=2
                ),
            )
        return img

    def calculate_angle(self, a, b, c):
        """For calculating angle between three landmarks"""
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return int(np.degrees(angle))

    def get_landmarks(self, h, w, c):
        """Extract landmark x- and y-coordinates and return as int numyp array"""
        if self.results and self.results.pose_landmarks:
            landmarks = self.results.pose_landmarks.landmark
            landmarks = [[lm.x, lm.y] for lm in landmarks]
            return np.multiply(landmarks, [w, h]).astype(int)
        else:
            return []

    def get_avg_visibility(self):
        if self.results and self.results.pose_landmarks:
            landmarks = self.results.pose_landmarks.landmark
            return np.mean([lm.visibility for lm in landmarks])
        else:
            return 0.0

    def is_skipping(self, lm):
        """calculate angles between hand, hips and shoulders"""
        left_angle = self.calculate_angle(lm[15], lm[11], lm[23])
        right_angle = self.calculate_angle(lm[16], lm[12], lm[24])
        avg_angle = (left_angle + right_angle) / 2

        if avg_angle > 22:
            return True
        else:
            return False

    def count_skipping(self, img, count, dir, is_skipping, base, position, draw=None):
        """
        The skipping is counted by checking if the position of the feet of
        the user cross a certain height threshold. This threshold is calculated by
        considering the relative height of the user.
        """
        h, w, c = self.img_shape
        lm = self.get_landmarks(h, w, c)

        # jump heights
        ########## ADJUST HERE FOR DIFFERNET JUMP HEIGHTS!!! #############
        distance_1_2 = int(h * 0.020)
        distance_1_3 = int(h * 0.097)

        # check if Person and Pose is detected
        if len(lm) != 0:

            # get highest foot landmark
            l_foot = lm[27]
            r_foot = lm[28]
            feet = l_foot[1] if l_foot[1] > r_foot[1] else r_foot[1]

            # get height of torso as factor: hips - shoulders
            hips = (lm[23][1] + lm[24][1]) / 2
            shoulders = (lm[11][1] + lm[12][1]) / 2
            height_Factor = hips - shoulders
            height_Factor = np.interp(height_Factor, (0.2, 0.4), (0.8, 1.2))

            anchor = feet

            if self.get_avg_visibility() > 0.8 and self.is_skipping(lm):
                if not is_skipping:
                    base = anchor
                    distance_1_2 = distance_1_2 * height_Factor
                    distance_1_3 = distance_1_3 * height_Factor

                is_skipping = True

                # Skipping-Counter mechanism
                line2 = base - distance_1_2
                line3 = base - distance_1_3

                if anchor < line2:
                    if position == 0:  # first time over line 2
                        count += 1
                        position = 1
                if anchor > line2:
                    if position == 1:
                        position = 0
                if anchor < line3:
                    if position == 1:  # first time over line 3
                        count += 1
                        position = 2
                if anchor > line3:  # double jump
                    if position == 2:
                        position = 1

            else:
                is_skipping = False
                base = anchor
        else:
            base = 0
            hips = 0
            shoulders = 0

        if draw == "jump":
            img = self.show_jump_lines(
                self, img, lm, base, distance_1_2, distance_1_3, w, count
            )
        elif draw == "angle":
            img = self.show_arm_angle(img, lm)

        return img, count, dir, is_skipping, base, position

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

    def show_arm_angle(self, img, lm):
        # show angle circle
        for i in [[16, 12, 24], [23, 11, 15]]:
            # get start angle
            # create new point
            new_lm = [lm[i[1]][0], lm[i[0]][1]]
            start_angle = self.calculate_angle(lm[i[0]], lm[i[1]], new_lm) + 90  # grad
            angle = self.calculate_angle(lm[i[0]], lm[i[1]], lm[i[2]])
            end_angle = start_angle - angle

            cv2.ellipse(
                img,
                center=(lm[i[1]][0], lm[i[1]][1]),
                axes=(130, 130),
                angle=0,
                startAngle=start_angle,
                endAngle=end_angle,
                color=(0, 255, 0) if self.is_skipping(lm) else (0, 0, 255),
                thickness=-1,
            )

            # print angle
            text_pos = np.mean([lm[j] for j in i], axis=0).astype(int)

            # setup text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str("{}".format(angle))
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(
                img,
                text,
                (text_pos[0] - (textsize[0] // 2), text_pos[1]),
                font,
                1,
                (255, 255, 255),
                3,
                2,
            )

        # show lm
        lm_indices = [11, 12, 15, 16, 23, 24]
        for i in lm_indices:
            cv2.circle(img, (lm[i][0], lm[i][1]), 8, (0, 0, 255), -1)

        # show lines
        lines = [[11, 23], [11, 15], [12, 24], [12, 16]]
        for l in lines:
            cv2.line(
                img,
                (lm[l[0]][0], lm[l[0]][1]),
                (lm[l[1]][0], lm[l[1]][1]),
                (0, 0, 255),
                3,
            )

        return img

    def show_jump_lines(self, img, lm, base, distance_1_2, distance_1_3, w, count):
        l_foot = lm[27]
        r_foot = lm[28]

        # cv2.line(img, (0, base - 10), (w, base), (0, 0, 255), 3)
        first = int(base - distance_1_2)
        cv2.line(img, (0, first), (w, first), (0, 0, 255), 3)
        second = int(base - distance_1_3)
        cv2.line(img, (0, second), (w, second), (0, 0, 255), 3)

        # show feet landmarks
        feet_color = (
            [(0, 255, 0), (255, 255, 255)]
            if l_foot[1] < r_foot[1]
            else [(255, 255, 255), (0, 255, 0)]
        )
        cv2.circle(img, (l_foot[0], l_foot[1]), 8, feet_color[0], -1)
        cv2.circle(img, (r_foot[0], r_foot[1]), 8, feet_color[1], -1)

        # put count
        x = 400
        y = 680
        cv2.rectangle(img, (x - 20, y - 60), (x + 100, y + 20), (0, 0, 0), -1)
        cv2.putText(
            img,
            str(count),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
            2,
        )

        return img


class SkipCounter:
    def __init__(self):
        self.detector = PoseDetector()
        self.vidcap = cv2.VideoCapture(0)# "videos/skipping_standard.mp4"
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

        frame = self.detector.find_pose(frame, draw=True)
        (
            frame,
            self.count,
            self.dir,
            self.is_skipping,
            self.baseline,
            self.position,
        ) = self.detector.count_skipping(
            frame,
            self.count,
            self.dir,
            self.is_skipping,
            self.baseline,
            self.position,
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
        print(fps_str)

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
