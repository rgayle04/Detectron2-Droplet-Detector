import sys
import random
import cv2
import os
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron_converter import convert_circle_csv_to_detectron2

train_name = "droplets_train"
val_name = "droplets_val"
ds_names = [train_name, val_name]

csv_path = sys.argv[1]
image_root = sys.argv[2]

def get_dataset():
    full_set = convert_circle_csv_to_detectron2(csv_path, image_root)
    random.seed(42)
    random.shuffle(full_set)
    split_index = int(0.8 * len(full_set))
    return full_set[:split_index], full_set[split_index:]

def main():
    for ds_name in ds_names:
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)

    train_dicts, val_dicts = get_dataset()
    DatasetCatalog.register(train_name, lambda: train_dicts)
    MetadataCatalog.get(train_name).set(thing_classes=["water_droplet"], evaluator_type="coco")
    DatasetCatalog.register(val_name, lambda: val_dicts)
    MetadataCatalog.get(val_name).set(thing_classes=["water_droplet"], evaluator_type="coco")

    # Visualize one random sample
    d = random.choice(train_dicts)
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(train_name))
    out = visualizer.draw_dataset_dict(d)

     # Config setup
    cfg = get_cfg()
    cfg.merge_from_file('C:\\Users\\rggayle\\detectron2\\configs\\COCO-InstanceSegmentation\\mask_rcnn_R_50_FPN_3x.yaml')
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.STEPS = (3000, 7000)
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.AMP.ENABLED = True
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = 100
    cfg.OUTPUT_DIR = "./output/exp_droplets_r50"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    # Evaluate
    evaluator = COCOEvaluator(val_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, val_name)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))



    cv2.imshow("Verification", out.get_image()[:, :, ::-1])
    cv2.resizeWindow('Verification', 400, 300)
    if cv2.waitKey(0) == 27:
        print("Closing...")
        exit()
    cv2.destroyAllWindows()

   

if __name__ == "__main__":
    main()
