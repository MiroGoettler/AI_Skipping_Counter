import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import datetime


def get_cmap_color(rgb_colors, max, val):
    if max == 0:
        max = 1
    colors = np.asarray(rgb_colors) / 255  # bring between 0 and 1
    cm = LinearSegmentedColormap.from_list("Custom", colors, N=100)
    return "rgb" + str(tuple((np.asarray(cm(val / max)) * 255).astype(int)[:-1]))


def seconds_to_min(sec):
    return time.strftime("%M:%S", time.gmtime(sec))


def format_seconds(sec, annotation=None):
    time_str = seconds_to_min(sec) + "min"
    if annotation:
        return f"{annotation}: <br>" + time_str
    else:
        return time_str


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        self._numFrames += 1

    def elapsed(self):
        now = datetime.datetime.now()
        return (now - self._start).total_seconds()

    def fps(self):
        return self._numFrames / self.elapsed()
