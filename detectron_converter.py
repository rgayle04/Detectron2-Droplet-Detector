import pandas as pd
import cv2
import numpy as np
import sys
import os 
import math
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def circle_to_polygon(cx, cy, r, max_error=0.5):
    if max_error >= r:
        num_points = 4
    else:
        theta = 2 * math.acos(1 - max_error/r)
        num_points = max(4, math.ceil(2 * math.pi / theta))
    theta = np.linspace(0, 2*np.pi, num_points)
    
    polygon = []
    
    for t in theta:
        x = cx + r * np.cos(t)
        y = cy + r * np.sin(t)
        polygon.extend([x,y])
    
    return [polygon]

def convert_circle_csv_to_detectron2(csv_path, image_root):
    cols = ["file_name", "x1", "y1", "r1", "x2", "y2", "r2"]
    df = pd.read_csv(csv_path, header=None, names=cols, sep="\t" if "\t" in open(csv_path).readline() else ",")

    dataset_dicts = []

    for idx, row in df.iterrows():
        file_path = os.path.join(image_root, row["file_name"])
        img = cv2.imread(file_path)
        height, width = img.shape[:2]

        annotations = []
        # First droplet
        x1, y1, r1 = float(row["x1"]), float(row["y1"]), float(row["r1"])
        bbox1 = [x1 - r1, y1 - r1, 2 * r1, 2 * r1]
        seg1 = circle_to_polygon(x1, y1, r1)
        annotations.append({
            "bbox": bbox1,
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": seg1,
            "category_id": 0,
        })
        # Second droplet
        x2, y2, r2 = float(row["x2"]), float(row["y2"]), float(row["r2"])
        bbox2 = [x2 - r2, y2 - r2, 2 * r2, 2 * r2]
        seg2 = circle_to_polygon(x2, y2, r2) 
        annotations.append({
            "bbox": bbox2,
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": seg2,
            "category_id": 0,
        })

        record = {
            "file_name": file_path,
            "image_id": idx,
            "height": height,  
            "width": width,   
            "annotations": annotations,
        }
        # If you have more than two droplets, you can extend the annotations list similarly
        

        dataset_dicts.append(record)

    return dataset_dicts

def save_detectron2_dataset(dataset_dicts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset_dicts, f, indent=4)



csv_path = sys.argv[1]  # Path to the input CSV file
image_root = sys.argv[2]  # Path to the directory containing images
output_json_path = sys.argv[3]  # Path to save the output JSON file

dataset_dicts = convert_circle_csv_to_detectron2(csv_path, image_root)

print(f"[INFO] Converted {len(dataset_dicts)} images to Detectron2 format.")

save_detectron2_dataset(dataset_dicts, output_json_path)


def get_dataset():
    return list[dataset_dicts]

#register_coco_instances('droplets_train', {}, "D:\Training Data\\base\detectron2-output\\annotations.json", "D:\Training Data\\base\\fr")
#DatasetCatalog.register("droplets_train", get_dataset)

#if MetadataCatalog.get("droplets_train"):
#    print("Dataset Registered: Success!")
#else:
#    print("Error Dataset not Registered!")
#.set(thing_classes=["droplet"])
# Metadata for the dataset