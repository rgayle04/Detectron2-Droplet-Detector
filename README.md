# Detectron2-Water-Droplet-Detector
Project made to take in a video of water droplets then outputs a csv containing info on water droplets using detectron2 as a base model

detectron2- env set up:
1: Download attached yaml file in repo
2: conda env create -f detectron2-env.yaml
3: conda activate detectron2
4: pip uninstall torch torchvision torchaudio
5: pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
6: conda install -c conda-forge opencv
  -pip install pandas
  -pip install numpy 
  -pip install opencv-python
  (There maybe be more depencies missing to use the detector) 



Dataset Registry/Training: 
python dataset_register.py "path to csv" "path to images" "output json path" 
[if cv2 shows up hit esc otherwise training may not go through] 
Eg. python dataset_register.py "D:\Training Data\base\fr\master.csv" "D:\Training Data\base\fr" "D:\Training Data\base\detectron2-output\annotations.json"


Detectron2 Prediction:

python infer_trained_model.py "path to input video" "path to output directory" "number of frames to skip(optional)"
Eg. python infer_trained_model.py "D:\Training Data\Droplet Videos\WP 30C DSC 1 to 1 to 0.2 SQE 0.01mg SER 189mOsm012.mp4" "D:\Training Data\base\detectron2-output" 10

Progress on a file to visualize the results from the csv is being made

