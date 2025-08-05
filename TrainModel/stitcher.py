import cv2
import os
import glob
import numpy as np
import sys
from pathlib import Path

# Define frame size (width, height)
frame_width, frame_height = 600, 600
frame_size = (frame_width, frame_height)

video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.wmv', '*.flv', '*.webm']

vpath1 = sys.argv[1]  # Video dir 1
vpath2 = sys.argv[2]  # Video dir 2
opath = sys.argv[3]   # Output dir

os.makedirs(opath, exist_ok=True)

def Fram_connect(frame1, frame2, h, w):
    if frame1 is None or frame2 is None:
        return None

    frame1 = cv2.resize(frame1, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    frame2 = cv2.resize(frame2, (int(w), int(h)), interpolation=cv2.INTER_AREA)

    combined = np.zeros((int(h), int(w * 2), 3), dtype=np.uint8)
    combined[:, :w] = frame1
    combined[:, w:] = frame2
    return combined


invideopaths1, invideopaths2 = [], []

if os.path.isdir(vpath1) and os.path.isdir(vpath2):
    for ext in video_extensions:
        invideopaths1.extend(glob.glob(os.path.join(vpath1, ext)))
        invideopaths2.extend(glob.glob(os.path.join(vpath2, ext)))

    print("Videos from Dir 1:", invideopaths1)
    print("Videos from Dir 2:", invideopaths2)

    for path1 in invideopaths1:
        name1 = Path(path1).stem
        for path2 in invideopaths2:
            name2 = Path(path2).stem
            if name1 == name2:
                print(f"Stitching {name1}...")

                cap1 = cv2.VideoCapture(path1)
                cap2 = cv2.VideoCapture(path2)

                if not cap1.isOpened() or not cap2.isOpened():
                    print(f"Failed to open one of the videos: {name1}")
                    continue

                output_path = os.path.join(opath, name1 + '_stitched.avi')
                vid = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width * 2, frame_height))

                while True:
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()

                    if not ret1 or not ret2 or frame1 is None or frame2 is None:
                        break

                    combined = Fram_connect(frame1, frame2, frame_height, frame_width)
                    if combined is None:
                        break

                    vid.write(combined)

                    # Optional: display
                    # cv2.imshow("Combined", combined)
                    # if cv2.waitKey(1) & 0xFF == 27:
                    #     break

                cap1.release()
                cap2.release()
                vid.release()
                print(f"Saved: {output_path}")

else:
    print("One or both input directories are invalid.")

    '''
    #Source of frame stitching code 
    https://karobben.github.io/2021/04/10/Python/opencv-v-paste/
    '''
