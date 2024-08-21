import numpy as np
import mss
import onnxruntime
import pyautogui
from PIL import Image
import os
import time

from dbd.utils.directkeys import PressKey, ReleaseKey, SPACE

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_monitor_attributes():
    width, height = pyautogui.size()
    object_size = 224

    monitor = {"top": height // 2 - object_size // 2,
               "left": width // 2 - object_size // 2,
               "width": object_size,
               "height": object_size}

    return monitor

def screenshot_to_numpy(screenshot):
    img = np.array(screenshot, dtype=np.float32)
    img = np.flip(img[:, :, :3], 2)
    img = img / 255.0
    img = (img - MEAN) / STD

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    img = np.float32(img)
    return img


if __name__ == '__main__':
    save_images = False  # if True, save detected great skill check images in saved_images/
    debug_monitor = False  # if True, check the image saved_images/monitored_image.png

    # Get monitor attributes to grab frames
    monitor = get_monitor_attributes()

    # Trained model
    filepath = "model.onnx"
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name

    img_folder = "saved_images"
    if (save_images or debug_monitor) and not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    with mss.mss() as sct:
        print("Monitoring the screen...")

        detected_images_idx = 0
        detected_images_idx_debug = 0
        last_pred_1_time = None
        last_interval = 0
        intervals = []
        sliding_window_size = 4  # Size of the sliding window
        
        while True:
            screenshot = sct.grab(monitor)
            img = screenshot_to_numpy(screenshot)

            # To check the monitor settings
            if debug_monitor:
                image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                image.save(os.path.join(img_folder, "monitored_image.png"))

            ort_inputs = {input_name: img}
            ort_outs = ort_session.run(None, ort_inputs)
            pred = np.argmax(np.squeeze(ort_outs, 0))
            
            current_time = time.time()            
            if pred == 1:
                if last_pred_1_time is not None:
                    interval = current_time - last_pred_1_time
                    intervals.append(interval)
                    if len(intervals) > sliding_window_size:
                        intervals.pop(0)
                print("pred", pred)
            elif pred == 2:
                PressKey(SPACE)
                ReleaseKey(SPACE)
                time.sleep(0.5)
                print("pred", pred)
                intervals.clear()  # Reset intervals
                last_pred_1_time = None  # Reset state
                last_interval = 0
                if save_images:
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    img.save(os.path.join(img_folder, "{}.png".format(detected_images_idx)))
                    detected_images_idx += 1
                    detected_images_idx_debug = 0
            
            # Compute the max interval
            # if len(intervals) == sliding_window_size and last_pred_1_time is not None:
            if last_pred_1_time is not None:
                max_interval = max(intervals)
                interval = current_time - last_pred_1_time
                diff_max_interval = interval - max_interval
                if abs(diff_max_interval) > 0.001:
                    print("Average interval", max_interval, "current_interval", interval, "last_interval", last_interval, "diff", interval - last_interval)
                    # PressKey(SPACE)
                    # ReleaseKey(SPACE)
                    # time.sleep(1)
                    print("Pressed space due to positive diff_max_interval:", diff_max_interval)
                    # intervals.clear()  # Reset intervals
                    # last_pred_1_time = None  # Reset state
                    # last_interval = 0
                    intervals.append(interval)
                    if len(intervals) > sliding_window_size:
                        intervals.pop(0)
                    if save_images:
                        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                        img.save(os.path.join(img_folder, "{}_{}_{}.png".format(detected_images_idx, detected_images_idx_debug, diff_max_interval)))
                        detected_images_idx_debug += 1
                last_interval = interval
            
            if pred == 1:
                last_pred_1_time = current_time