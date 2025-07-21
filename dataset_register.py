from detectron2.data import DatasetCatalog, MetadataCatalog
import random
import cv2
import os 
from detectron_converter import convert_circle_csv_to_detectron2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


ds_name = "droplets_train"

def get_dataset():
    return convert_circle_csv_to_detectron2(
        r"D:\\Training Data\base\\fr\\master.csv",
        r"D:\\Training Data\base\\fr"
    )

def main():
    # Remove old registration if any
    if ds_name in DatasetCatalog.list():
        DatasetCatalog.remove(ds_name)

    DatasetCatalog.register(ds_name, get_dataset)
    MetadataCatalog.get(ds_name).set(thing_classes=["water_droplet"])

        #Commment out unless trying to verify Dataset registration 
    #print("Registered datasets:", DatasetCatalog.list())

    dataset_dicts = DatasetCatalog.get(ds_name)  # This should now call get_dataset() and return list of dicts
    print(f"Loaded {len(dataset_dicts)} samples")

    d = random.choice(dataset_dicts)
    img = cv2.imread(d["file_name"])

    visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(ds_name))
    out = visualizer.draw_dataset_dict(d)

    cv2.imshow("Verification", out.get_image()[:, :, ::-1])
    cv2.resizeWindow('Verification', 400, 300)
    key = cv2.waitKey(0)
    
    if key == 27:
        print("Closing...")
        exit()

    cv2.destroyAllWindows()

    cfg = get_cfg()
    cfg.merge_from_file('C:\\Users\\rggayle\detectron2\configs\COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml')
    cfg.DATASETS.TRAIN = ("droplets_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD=100

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
