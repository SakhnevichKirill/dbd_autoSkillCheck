import numpy as np
import cv2
import mss
import time
import os

from dbd.utils.frame_grabber import get_monitor_attributes


if __name__ == '__main__':
    # Make new dataset folder, where we save the frames
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dataset_folder = os.path.join("dataset", timestr)
    os.mkdir(dataset_folder)

    # Get monitor attributes
    monitor = get_monitor_attributes()

    with mss.mss() as sct:
        i = 0

        # Infinite loop
        while True:
            screenshot = np.array(sct.grab(monitor))
            cv2.imwrite(os.path.join(dataset_folder, "{}_{}.png".format(timestr, i)), screenshot)
            i += 1
