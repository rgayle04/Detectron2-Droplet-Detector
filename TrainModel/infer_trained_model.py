import cv2
import os
#import imageio
import numpy as np
import sys
import torch
import pandas as pd
from pathlib import Path
from detectron2.utils.visualizer import ColorMode, Visualizer 
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import time
from utils import getModelInput, processframe, getcb2cImg, getBoundingSquare
#tracker = Sort()

#track_colors = defaultdict(lambda: tuple(np.random.randint(0,255, size=3).tolist()))

# --- Detectron2 Setup ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('./COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')) #may need  to be changed for other users
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # For 'droplet' class
cfg.TEST.EVAL_PERIOD = 100
predictor = DefaultPredictor(cfg)
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),ColorMode.IMAGE_BW)

# --- Metadata Setup ---
MetadataCatalog.get('droplets_train').thing_classes = ['droplet']
MetadataCatalog.get('droplets_train').thing_colors = [(0, 255, 0)]

#premptive setup
g_pixelsPerUnit = 6.12
g_frameskip = 1

g_currentframe = 1

basic_frame_measurements = []
#complete_droplet_Measurements = []


# --- Input Arguments ---
vpath = sys.argv[1]

opath = sys.argv[2]

frameskip = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# --- Video Setup ---
cap = cv2.VideoCapture(vpath)

fps = cap.get(cv2.CAP_PROP_FPS)

output_video_path = os.path.join(opath, Path(vpath).stem + '.avi')

csvname = os.path.join(opath, Path(vpath).stem + '.csv')
print(f'csv out: {csvname}\nvideo out: {opath}\n')
print(f"{vpath} at FPS: {fps}\n")

outfile = open(csvname, 'w')
header = ("Time Stamp,Droplet 1 Radius,Droplet 1 Volume,"
              "Droplet 2 Radius,Droplet 2 Volume,Total Volume,"
              "DIB Radius,Contact Angle,Radial Distance,\n")
outfile.write(header)


def draw_fixed_color_instances(frame, instances, colors):
    
    droplet_circles = []
    
    frame_drawn = frame.copy()
    if not instances.has("pred_masks"):
        return frame_drawn

    pred_masks = instances.pred_masks.cpu().numpy()
    boxes = instances.pred_boxes.tensor.cpu().numpy().astype(int)
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    try:
            image = cv2.GaussianBlur(frame_drawn, (5, 5), 0)
    except:
        exit()


    (h,w) = image.shape[:2]

    #droplet_data = []

    for i in range(len(pred_masks)):
        color = colors[i % len(colors)]
        mask = pred_masks[i]
        #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #if contours and len(contours[0])> 5:
          #  area = cv2.contourArea(contours[0])
          #  droplet_data.append((i,area))

        colored_mask = np.zeros_like(frame_drawn, dtype=np.uint8) #highlights area of the droplets with masks which are then used to get the data from the droplets later on 
        colored_mask[mask] = color

        frame_drawn = cv2.addWeighted(frame_drawn, 1.0, colored_mask, 0.5, 0)

        x1, y1, x2, y2 = boxes[i]
        #cx1, c2y1, c2x2, c2x2 = boxes[i+1]
        
#        bsx1, bsy1, bsx2, bsy2 = getBoundingSquare((bsx1, bsy1, bsx2, bsy2),h,w)

#        (bh, bw) = (bsy2-bsy1,bsx2-bsx1)

#        cibox= [(w/2)-(bw/2), (h/2)-(bh/2), (w/2)+(bw/2), (h/2)+(bh/2)]
#        cibox = list(map(lambda x: int(x), cibox))

        

        #preds= predictor(cmi)
        #c1x1, c1y1, c1x2, c1y2 = getBoundingSquare((c1x1,c1y1,c1x2,c1y2), h, w)

        #c2x1, c2y1, c2x2, c2y2 = getBoundingSquare((c2x1, c2y1, c2x2, c2y2), h, w)
        #circleimage = np.zeros((h,w,3), np.uint8)
        #circleimage[:]=(255,255,255)
        #(bh, bw) = (y2-y1,x2-x1)
        #cibox=[(w/2)-(bw/2), (h/2)-(bh/2), (w/2)+(bw/2),(h/2)+(bh/2)]
        #cibox= list(map(lambda x: int(x), cibox))
        #circleimage[cibox[1]:cibox[3], cibox[0]:cibox[2]] = image[y1:y2, x1:x2].copy()

        #cm_input = getModelInput(circleimage, 224)
        #preds = predictor(cm_input)
        #(c1x1,c1y1)



        

        label_text = f'Class {classes[i]}: {scores[i]:.2f}'
        cv2.putText(frame_drawn, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        #repetition of data issue starts in these loops
        #fixed -- 7/9 
        #droplet_data = sorted(key=area(lambda x: droplet_data[1]))


    for i, box in enumerate(boxes):
        
        x1, y1, x2, y2 = box  
        #print(x1,y1,x2,y2)
        x1, y1, x2, y2 = getBoundingSquare((x1,y1,x2,y2), h, w)
        
        mask = pred_masks[i].astype('uint8')

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame_drawn, contours, -1,(0,255,0),2) #double checks the shape of droplets to ensure the measurements given are accurate 
        
        (cx,cy), cr = cv2.minEnclosingCircle(contours[0])
        cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0,0,255),2) #displays the bounding boxes of the droplets when predicting on videos 

        #if cr < 1 or cr > 1000:
         #   print(f'Warning: Invalid circle radius {cr} at frame {g_currentframe}')
         #   continue

       # if not (0<=cx < frame_width and 0 <=cy < frame_height):
        #    print(f'Warning: Circle center out of bounds ({cx},{cy}) at frame {g_currentframe}')
        #   continue
        droplet_circles.append((cx, cy, cr))

        droplet_circles = sorted(droplet_circles, key=lambda c: c[0]) #stops issue of circles being "swapped"

        #print(f'Droplet circles: {droplet_circles}')
    for i in range(len(droplet_circles)-1):

        c1 = list(map(lambda x: int(x), droplet_circles[i]))
        c2 = list(map(lambda x: int(x), droplet_circles[i+1]))
        
        #comment out here or in frame processing for debugging 
        #print(f'Circle 1: {c1}')
        #print(f'Circle 2: {c2}')

        result = processframe(c1, c2)
        #also ensures the cx, cy and cr are accurate to the actual droplets 
        cv2.circle(frame_drawn, (c1[0],c1[1]), c1[2], (0, 255, 0 ), 2)
        cv2.circle(frame_drawn, (c2[0],c2[1]), c2[2], (0, 255, 0), 2)


        droplet_circles.clear() #prevents repetition of data in loop  
        if result is not None: 
            r1, v1, r2, v2, tv, rdib, theta_deg, lr = result
            timestamp = float(g_currentframe)/fps
            outfile.write(f'{timestamp},{r1},{v1},{r2},{v2},{tv},{rdib},{theta_deg},{lr}\n')
            outfile.flush()
        else:
            print(f'[Warning] No result from processframe() for frame {g_currentframe}')


    return frame_drawn

frame_size=(800,800)
vid = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)
fixed_colors = [(255, 0, 0), (255, 0, 0)]  # Blue and Green for first two droplets (keeping both as blue for now)

print(f'Reading {vpath} video')

#while loop set to iterate through each frame till end of video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #frame = cv2.resize(frame,(600,600))
    if g_currentframe % frameskip != 0:
        g_currentframe += 1
        continue
    
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")
    #print(instances)

    
    frame = draw_fixed_color_instances(frame, instances, fixed_colors)
    frame=cv2.resize(frame, frame_size)
    #frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #frame= cv2.resize(frame, (1300,1300))
    
    
    cv2.imshow("RT Detection", frame)
    
    #cv2.resizeWindow("RT Detection", 400, 300)
    
    vid.write(frame)
    
    key = cv2.waitKey(10)
    #be careful with this cause once 'esc' is hit immediately stops the prediction/processing 
    if key == 27:
        print("Closing...")
        exit()
    g_currentframe += 1

print(f'Writing to {csvname}')
vid.release()
outfile.close()


cap.release()
cv2.destroyAllWindows()

'''


   '''
