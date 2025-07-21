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
cfg.merge_from_file(model_zoo.get_config_file('./COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
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
droplet_circles = []
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
print(f"{vpath} at FPS: {fps}")

outfile = open(csvname, 'w')
header = ("Time Stamp,Droplet 1 Radius,Droplet 1 Volume,"
              "Droplet 2 Radius,Droplet 2 Volume,Total Volume,"
              "DIB Radius,Contact Angle,Radial Distance,\n")
#pd.DataFrame(columns=columns).to_csv(csvname, index=False)
outfile.write(header)



#outfile = open(csvname,'w')

#header =("Time Stamp,Droplet 1 Radius,Droplet 1 Volume,"         "Droplet 2 Radius,Droplet 2 Volume,Total Volume,"        "DIB Radius, Contact Angle, Radial Distance,\n")


def draw_fixed_color_instances(frame, instances, colors):

    frame_drawn = frame.copy()
    if not instances.has("pred_masks"):
        return frame_drawn

    masks = instances.pred_masks.cpu().numpy()
    boxes = instances.pred_boxes.tensor.cpu().numpy().astype(int)
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    try:
            image = cv2.GaussianBlur(frame, (5, 5), 0)
    except:
        exit()


    (h,w) = image.shape[:2]

    for i in range(len(masks)):
        color = colors[i % len(colors)]
        mask = masks[i]

        colored_mask = np.zeros_like(frame_drawn, dtype=np.uint8)
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



        cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), color, 2)

        label_text = f'Class {classes[i]}: {scores[i]:.2f}'
        cv2.putText(frame_drawn, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        #repetition of data issue starts in these loops
        #fixed -- 7/9 

    #droplet_circles = []
    #basic_frame_measurements = []
    #complete_droplet_Measurements = []

    for i, box in enumerate(boxes):
        #print(box)
        x1, y1, x2, y2 = box  
        #print(x1,y1,x2,y2)
        x1, y1, x2, y2 = getBoundingSquare((x1,y1,x2,y2), h, w)
        #x1 = int(x1*w)
        #y1 = int(y1*h)
        #x2 = int(x2*w)
        #y2 = int(y2*h)
        #c1bsqimg = getcb2cImg(image, )

        mask = masks[i].astype('uint8')

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame, contours, -1,(0,255,0),2)
        
        (cx,cy), cr = cv2.minEnclosingCircle(contours[0])

        #if cr < 1 or cr > 1000:
         #   print(f'Warning: Invalid circle radius {cr} at frame {g_currentframe}')
         #   continue

       # if not (0<=cx < frame_width and 0 <=cy < frame_height):
        #    print(f'Warning: Circle center out of bounds ({cx},{cy}) at frame {g_currentframe}')
        #   continue

        droplet_circles.append((cx, cy, cr))
        #cx = (x1 + x2) / 2
        #cy = (y1 + y2) / 2

        #cr = (w + h) / 4

        #droplet_circles.append((cx, cy, cr))
        '''basic_frame_measurements.append({
            "frame": g_currentframe,
            "droplet_id": i,
            "x": cx,
            "y": cy,
            "radius_pixels": cr,
            "radius_units": cr / g_pixelsPerUnit
        })'''

    

    for i in range(len(droplet_circles)-1):

        c1 = list(map(lambda x: int(x), droplet_circles[i]))
        c2 = list(map(lambda x: int(x), droplet_circles[i+1]))
        #c1 = droplet_circles[i]
        #c2 = droplet_circles[i+1]
        #c1[0] += x1
        #c1[1] += y1
        #c2[0] += x2
        #c2[1] += y2

        
        #cv2.circle(frame, (c1[0],c1[1]), c1[2], (0, 255, 0 ), 2)
        #cv2.circle(frame, (c2[0],c2[1]), c2[2], (0, 255, 0), 2)

        #print((c1[0],c1[1]), c1[2], 'Circle1 Coor')

        #print((c2[0],c2[1]), c2[2], 'Circle2 Coor')

        

        cv2.circle(frame_drawn, (c1[0],c1[1]), c1[2], (0, 255, 0 ), 2)
        cv2.circle(frame_drawn, (c2[0],c2[1]), c2[2], (0, 255, 0), 2)

        result = processframe(c1, c2)

        droplet_circles.clear()
        if result is not None:
            r1, v1, r2, v2, tv, rdib, theta_deg, lr = result
            timestamp = float(g_currentframe)/fps
            outfile.write(f'{timestamp},{r1},{v1},{r2},{v2},{tv},{rdib},{theta_deg},{lr}\n')
            outfile.flush()
        else:
            print(f'[Warning] No result from processframe() for frame {g_currentframe}')


   # df_pairs = pd.DataFrame(complete_droplet_Measurements)
   # complete_droplet_Measurements = []
   # # df_pairs.to_csv(csvname, index=False)

    # df = pd.read_csv(csvname)
    # df=df.drop_duplicates()
    # df.to_csv(csvname, index=False)
    # score = float(scores[i])
    # label = int(classes[i]) if classes is not None else -1

    # print(score)

    return frame_drawn

#writer = imageio.get_writer(output_video_path, fps=fps)
frame_size=(400,400)
vid = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)
fixed_colors = [(255, 0, 0), ( 255, 0, 0)]  # Green and Blue for first two droplets

print(f'Reading {vpath}')

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

    #frame = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #frame= cv2.resize(frame, (1300,1300))
    #frame=cv2.resize(frame, frame_size)
    
    cv2.imshow("RT Detection", frame)
    
    #cv2.resizeWindow("RT Detection", 400, 300)
    
    vid.write(frame)
    
    key = cv2.waitKey(10)
    if key == 27:
        print("Closing...")
        exit()
    g_currentframe += 1

print(f'Writing to {csvname}')
vid.release()
outfile.close()


#df_pairs = pd.DataFrame(complete_droplet_Measurements)
#df_pairs.drop_duplicates(inplace=True)
#df_pairs.to_csv(csvname, index=False)
#complete_droplet_Measurements.clear()


#outfile.close()

cap.release()
cv2.destroyAllWindows()

'''


    (bbx1, bbx2, bby1, bby2)=boxes[0]

    image = cv2.GaussianBlur(frame, (5, 5), 0)

    (h,w)=image.shape[:2]    

    bbx1 = int(bbx1 * w)
    bbx2 = int(bbx2 * w)
    bby1 = int(bby1 * h)
    bby2 = int(bby2 * h)

    bsx1, bsy1, bsx2, bsy2 = getBoundingSquare((bbx1, bby1, bbx2, bby2), h, w)

    circleimage = np.zeros((h,w,3), np.uint8)
    circleimage[:] = (255,255,255)
    (bh, bw)=(bsy2-bsy1, bsx2-bsx1)
    cibox = [(w/2)-(bw/2), (h/2)-(bh/2), (w/2)+(bw/2), (h/2)+(bh/2)]
    cibox = list(map(lambda x: int(x), cibox))
    circleimage[cibox[1]:cibox[3], cibox[0]:cibox[2]] = image[bsy1:bsy2, bsx1:bsx2].copy()
       vpath = sys.argv[1]


    opath = sys.argv[2]

    cap=cv2.VideoCapture(vpath)
    output_video_path = os.path.join(opath, Path(vpath).stem + '.avi')
    csvname = os.path.join(opath, Path(vpath).stem + '.csv')

    print(f'csv out: {csvname}\nvideo out: {opath}\n')

    print(f"FPS: {fps}")

    outfile = open(csvname,'w')

    header =("Time Stamp,Droplet 1 Radius,Droplet 1 Volume,"
         "Droplet 2 Radius,Droplet 2 Volume,Total Volume,"
         "DIB Radius, Contact Angle, Radial Distance,\n")
    outfile.write(header)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameskip = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    frame_size = (600, 600)

    #writer = imageio.get_writer(output_video_path, fps=fps, frame_size)
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)
    while cap.isOpened():
        return 

        df_pairs = pd.DataFrame(complete_droplet_Measuremments)
    df_pairs.to_csv(csvname, index=False)


                    complete_droplet_Measurements.append({
                    "timestamp": frame_index / frameskip,
                    "Droplet 1 Radius": r1,
                    "Droplet 1 Volume": v1,
                    "Droplet 2 Radius": r2,
                    "Droplet 2 Volume": v2,
                    "Total Volume": tv,
                    "DIB Radius": rdib,
                    "Contact Angle": theta_deg,
                    "Radial Distance": lr
                })

            if complete_droplet_Measurements:
                df_pairs=pd.DataFrame(complete_droplet_Measurements)
                df_pairs.to_csv(csvname, mode='a', header=False, index=False)
                complete_droplet_Measurements.clear()

    circleimage = np.zeros((h,w,3), np.uint8)
        circleimage[:]=(255,255,255)
        (bh, bw) = (y2-y1,x2-x1)
        bsw = int((x2-x1)/2)
        cibox=[(w/2)-(bw/2), (h/2)-(bh/2), (w/2)+(bw/2),(h/2)+(bh/2)]
        cibox= list(map(lambda x: int(x), cibox))
        circleimage[cibox[1]:cibox[3], cibox[0]:cibox[2]] = image[y1:y2, x1:x2].copy()
        
        cm_input = getModelInput(circleimage, 224)
        preds = predictor(cm_input)
        (cx1,cy1,cx2,cy2) = preds
        (cx1,cy1,cx2,cy2) = map(lambda x: int(x*w), (cx1,cy1,cx2,cy2))
        
        (cox, coy) = ((w/2)-bsw, (h/2-bsw))
        (tvx, tvy) = (x1-cox, y1-coy)
        (cx1, cx2) = map(lambda x: x+tvx, (cx1, cx2))
        (cy1, cy2) = map(lambda y: y + tvy, (cy1, cy2))




   ''' 