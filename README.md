# Detectron2-Water-Droplet-Detector

## Anaconda Environment set up:

Download miniconda from [here](https://docs.anaconda.com/products/distribution/download/) and install as administrator. 

#
  1: Download attached yaml file in repo
  
  2: Open a miniconda prompt with administrator privelages in the repo and create the conda environment:
  
    conda env create -f detectron2-env.yaml
    conda activate detectron2
  
  4: Remove broken dependencies:
    
    pip uninstall torch torchvision torchaudio
  
  5: Install pytorch
    
    pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
  
  6: Install python dependencies
    
    conda install -c conda-forge opencv
    pip install pandas numpy opencv-python


#
## Dataset Registry/Training: 
registers a dataset within Detectron2 with a prexisting master csv, then trains a prexisting detectron2 model for image prediction results in a coco annotations csv and a trained model registered within built in output directory 

python dataset_register.py "path to csv" "path to images" "output json path" 


Eg. python dataset_register.py "D:\Training Data\base\fr\master.csv" "D:\Training Data\base\fr" "D:\Training Data\base\detectron2-output\annotations.json"

[To modify number of epochs use equation epoch = (max_iter * batch_size)/total # of images]

#
## Detectron2 Prediction:

performs image prediction on input videos then outputs a video of two droplets and an output csv with the radius, volume, contact angle, etc of both water droplets in video 

python predict_video.py "path to input video" "path to output directory" "number of frames to skip(optional)"

Eg. python predict_video.py "D:\Training Data\Droplet Videos\WP 30C DSC 1 to 1 to 0.2 SQE 0.01mg SER 189mOsm012.mp4" "D:\Training Data\base\detectron2-output" 10

Progress on a file to visualize the results from the csv is being made

#
## Video Stitcher:
simply will take two videos with matching stem names and will stitch them together for side by side, frame to frame comparison 

python stitcher.py "path to directory 1" "path to directory 2" "output directory"

Eg. python stitcher.py "E:\Training Data\base\output\output2" "E:\Training Data\base\detectron2-output" "E:\Training Data\base\testing output"


