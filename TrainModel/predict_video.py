import cv2
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from numpy.polynomial import polynomial as P 
import pandas as pd
import glob
import numpy as np
import sys
import scipy.stats as stats
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression 
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
#from deep-sort-realtime.deepsort_tracker import DeepSort
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from utils import getBoundingSquare, processframe  

# ---------------- Detectron2 Setup ----------------

cfg = get_cfg()
cfg.OUTPUT_DIR = "./output/exp_droplets_r50"
cfg.merge_from_file(model_zoo.get_config_file('./COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')) #may need  to be changed for other users not fully sure 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.TEST.EVAL_PERIOD = 100
predictor = DefaultPredictor(cfg)

#tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.3)

#droplet_history={}

MetadataCatalog.get('droplets_train').thing_classes = ['droplet']
MetadataCatalog.get('droplets_train').thing_colors = [(0, 255, 0)]
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE_BW)

video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.wmv', '*.flv', '*.webm']


def get_color_from_position(x, w):
    if x<=(w/2):
        return (255, 0,0)
    else: # Blue(1) and Green(2) for first two droplets 
        return (0,255,0)


def draw_fixed_color_instances(frame, instances, outfile, g_currentframe, fps):
    droplet_circles = []
    frame_drawn = frame.copy()
    if not instances.has("pred_masks"):
        return frame_drawn

    pred_masks = instances.pred_masks.cpu().numpy()
    boxes = instances.pred_boxes.tensor.cpu().numpy().astype(int)
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    try:
        image = cv2.GaussianBlur(frame_drawn, (5,5), 0)
    except:
        exit()
    (h, w) = image.shape[:2]

    for i in range(len(pred_masks)):
        mask = pred_masks[i]
        x1, y1, x2, y2 = boxes[i]
        cx = int((x1 + x2) / 2)
        color = get_color_from_position(cx, w)

        colored_mask = np.zeros_like(frame_drawn, dtype=np.uint8) #highlights area of the droplets with masks which are then used to get the data from the droplets later on 
        colored_mask[mask] = color
        frame_drawn = cv2.addWeighted(frame_drawn, 1.0, colored_mask, 0.5, 0)

        label_text = f'Class {classes[i]}: {scores[i]:.2f}'
        cv2.putText(frame_drawn, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for i, box in enumerate(boxes):
        #x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        #roi = image[y1:y2, x1:x2]
        #gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        #circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, )

        x1, y1, x2, y2 = getBoundingSquare(box, h, w)
        mask = pred_masks[i].astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        cv2.drawContours(frame_drawn, contours, -1, (0, 255, 0), 2) #double checks the shape of droplets to ensure the measurements given are accurate 
        (cx, cy), cr = cv2.minEnclosingCircle(contours[0])
        cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0, 0, 255), 2) #displays the bounding boxes of the droplets when predicting on videos 

        droplet_circles.append((cx, cy, cr))

    droplet_circles = sorted(droplet_circles, key=lambda c: c[0]) #stops issue of circles being "swapped" marks left most as droplet 1 and rightmost as droplet 2 

    if len(droplet_circles) == 2:
        c1 = list(map(int, droplet_circles[0]))
        c2 = list(map(int, droplet_circles[1]))

        #comment out here or in frame processing for debugging 
        #print(f'Circle 1: {c1}')
        #print(f'Circle 2: {c2}')


        result = processframe(c1, c2)

        #also ensures the cx, cy and cr are accurate to the actual droplet
        cv2.circle(frame_drawn, (c1[0], c1[1]), c1[2], (0, 255, 0), 2)
        cv2.circle(frame_drawn, (c2[0], c2[1]), c2[2], (0, 255, 0), 2)

        cv2.circle(frame_drawn, (c1[0], c1[1]), 2, (0, 0, 255), 2)
        cv2.circle(frame_drawn, (c2[0], c2[1]), 2, (0, 0, 255), 2)

        if result is not None:
            r1, v1, r2, v2, tv, rdib, theta_deg, lr = result
            timestamp = float(g_currentframe) / fps
            outfile.write(f'{timestamp},{r1},{v1},{r2},{v2},{tv},{rdib},{theta_deg},{lr}\n')

    return frame_drawn


def process_video(video_path, output_dir, frameskip=1):
    # --- Video Setup ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    name = Path(video_path).stem
    frame_size = (800, 800)

    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, f"{name}.avi")
    csv_path = os.path.join(output_dir, f"{name}.csv")

    print(f"{name} video")
    print(f"video out: {output_video_path}")
    print(f"csv out: {csv_path}\n")

    vid = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)
    with open(csv_path, 'w') as outfile:
        outfile.write("Time Stamp,Droplet 1 Radius,Droplet 1 Volume,Droplet 2 Radius,Droplet 2 Volume,"
                      "Total Volume,DIB Radius,Contact Angle,Radial Distance\n")

        g_currentframe = 1
        #while loop set to iterate through each frame till end of video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if g_currentframe % frameskip != 0:
                g_currentframe += 1
                continue

            outputs = predictor(frame)
            instances = outputs["instances"].to("cpu")

            frame = draw_fixed_color_instances(frame, instances, outfile, g_currentframe, fps)
            frame = cv2.resize(frame, frame_size)
            vid.write(frame)

            cv2.imshow("RT Detection", frame)
            if cv2.waitKey(1) == 27:
                print("Interrupted by user.")
                break
            g_currentframe += 1
    
    df = pd.read_csv(csv_path)

    if df.empty:
        return
    else:

        std = np.std(df['DIB Radius'], ddof= 1)
        print('DIB Rad Standard Deviation:', std)
        mean = df['DIB Radius'].mean()
        print('DIB Rad Mean:', mean)

        uL = mean + (std*3) 
        print(f'Upper Limit: {uL}') 
        
        lL = mean - (std*3)
        print(f'Lower Limit: {lL}')
        #df = df[df['DIB Radius'].notna()]

        df.dropna(subset=['DIB Radius'], inplace=True)  # Remove NaN rows first

        outliers = np.where((df['DIB Radius'] > uL) | (df['DIB Radius'] < lL))
        print(f'Rows before Cleaning: {len(df)}')
        print(f'Number of Outliers: {len(outliers[0])}')

        if len(outliers[0]) < len(df) * 0.5:  # Don't remove more than 50%
            df.drop(outliers[0], axis=0, inplace=True)
            df.reset_index(drop=True, inplace=True)  # Reset index after dropping
            print(f'Rows remaining: {len(df)}')
        else:
            print("Too many outliers detected - skipping removal")
        split_name = name.split(" ")
        
        sp_name = split_name[-1]

        osmP = ""

        for i in range(3):
            if len(osmP)<3:
                osmP+=sp_name[i]
        #print(int(osmP))
        
        osmP = float(osmP)/1000

        df['Org. Concentr'] = None

        df.at[0, 'Org. Concentr'] = osmP

        df['Adjusted Time'] = (df['Time Stamp']-df.loc[0, 'Time Stamp'])

        df['DIB Area'] =  math.pi*(df['DIB Radius']**2)
        
        av = df.loc[0, 'Droplet 1 Volume']
        
        df['(V/V0)^2']= (df['Droplet 1 Volume']/av)**2

        df['Linear Permebility Section:'] = None 

        df['Init Radius'] = None
        df.at[0, 'Init Radius'] = (df.loc[0,'Droplet 1 Radius']/10000)

        #f = (df.loc[1,'Droplet 1 Radius']/10000) 

        df['Init Vol'] = None
        df.at[0, 'Init Vol'] = (4/3)*3.1415*(df.loc[0,'Init Radius']**3)

        df['Linearized DIB Radius'] = None
        slope, intercept, r, p, std_error = stats.linregress(df['Adjusted Time'], df['DIB Radius'])

        df.at[0, 'Linearized DIB Radius'] = intercept

        df['DIB Radius(cm)'] = None
        df.at[0, 'DIB Radius(cm)'] = intercept/10000

        df['DIB Area(cm^2)'] = None
        df.at[0, 'DIB Area(cm^2)'] = math.pi*(df.loc[0,'DIB Radius(cm)']**2)

        slope, intercept, r, p, std_error = stats.linregress(df['Adjusted Time'], df['(V/V0)^2'])
        
        summary_cols = {
        'slope (V/V0)^2': slope,
        'r^2': intercept,
        'Permeability (avg DIB Rad)': ((slope/2) * df.loc[0, 'DIB Radius(cm)']) / (df.loc[0, 'DIB Area(cm^2)'] * 0.018 * df.loc[0, 'Org. Concentr']) * 2,
        '3rd Degree Polynomial Section:': None
        }

        # Only set these at row 0
        for col, val in summary_cols.items():
            if col not in df.columns:
                df[col] = np.nan  # Use NaN instead of None
            df.at[0, col] = val

        x = df['Adjusted Time'].dropna().values

        y = df['DIB Area'].dropna().values

        min_len = min(len(x), len(y))

        x = x[:min_len]
        y = y[:min_len]

        coeffs = P.polyfit(x,y,3)


        #X = np.column_stack([x**1, x**2, x**3])

        #X_with_intercept = np.column_stack([np.ones(len(x)), X])

        #coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

        
#        coefficients = np.polyfit(x,y,3)
#        a3, a2, a1, a0 = coefficients

        #a = coefficients.tolist() 
        df['A'] = None
        df.at[0, 'A'] = coeffs[3]
        df['B'] = None
        df.at[0, 'B'] = coeffs[2]
        df['C'] = None
        df.at[0, 'C'] = coeffs[1]
        df['D'] = None
        df.at[0, 'D'] = coeffs[0] 
        df['Eval']=((0.018*df.loc[0, 'Org. Concentr']*df.loc[0, 'A']*df['Adjusted Time']**4)/(2*df.loc[0, 'Droplet 1 Volume'])+(2*0.018*df.loc[0, 'Org. Concentr']*df.loc[0, 'B']*df['Adjusted Time']**3)/(3*df.loc[0,'Droplet 1 Volume'])+(0.018*df.loc[0, 'Org. Concentr']*df.loc[0, 'C']*df['Adjusted Time']**2)/df.loc[0,'Droplet 1 Volume']+(2*0.018*df.loc[0, 'Org. Concentr']*df.loc[0, 'D']*df['Adjusted Time'])/df.loc[0,'Droplet 1 Volume'])

        
        #df.at[0, 'Permeability'] = ((slope/2)* df.loc[0, 'DIB Radius(cm)'])/(df.loc[0, 'DIB Area(cm^2)']*0.018*(df.loc[0, 'Org. Concentr']))

        
        #slope, intercept, r, p ,std_error = stats.linregress(df['Eval'], df['(V/V0)^2'])
        
        x = df['Eval'].values
        y = df['(V/V0)^2'].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        
        slope, intercept = np.polyfit(x,y,1)
        df['Permeability (slope)']= None
        df.at[0, 'Permeability (slope)'] = slope

        df['Permeability (intercept)'] = None
        df.at[0, 'Permeability (intercept)'] = intercept

        df.to_csv(csv_path, index=False)
        

    cap.release()
    vid.release()
    cv2.destroyAllWindows()
    print(f"[DONE] {name} processed.\n")


def main(vpath, opath, frameskip=1):
    if os.path.isdir(vpath):
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(vpath, ext)))
        for video_file in video_files:
            process_video(video_file, opath, frameskip)
    else:
        process_video(vpath, opath, frameskip)


if __name__ == "__main__":
    # --- Input Arguments ---

    vpath = sys.argv[1]

    opath = sys.argv[2]

    frameskip = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    main(vpath, opath, frameskip=frameskip)

'''


   '''
