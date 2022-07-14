# A2S-v2

Our new work on Unsupervised Salient Object Detection (USOD) task.  
Paper and code will be published soon.  
Here are our [saliency maps](https://drive.google.com/drive/folders/15YOcPQ5vzBqlk50DEEVuBXSo5YzqpT89?usp=sharing) and [pseudo labels](https://drive.google.com/drive/folders/1agLC1iNoONw008jaqEvfRalrBWFWIAL4?usp=sharing).  
  
Our code is based on our [SOD benchmark](https://github.com/moothes/SALOD).  
Pretrained backbone: [MoCo-v2](https://github.com/facebookresearch/moco).  

 ## Dataset
 For convenience， we reorganized the datasets for different SOD tasks.  
 Some datasets have not been used in our paper, and you can use for future work. 
 ### RGB SOD
 2 train sets: ```DUTS-TR``` and ```MSB-TR``` (the train split of MSRA-B).  
 6 test sets: ```HKU-IS```, ```PASCAL-S```, ```ECSSD```, ```DUTS-TE```, ```DUT-OMRON``` and ```MSB-TE``` (the test split of MSRA-B).  
 ### RGB-D SOD
 2 train sets: ```RGBD-TR``` and ```RGBD-TR-2985```. RGBD-TR include 2185 images, the same as most RGB-D SOD methods.  
 9 test sets: ```DUT```, ```LFSD```, ```NJUD```, ```NLPR```, ```RGBD135```, ```SIP```, ```SSD```, ```STERE1000``` and ```STEREO```.  
 ### RGB-T SOD
 1 train set: ```VT5000-TR```, which include 2500 train images in VT5000.  
 3 test sets: ```VT821```, ```VT1000``` and ```VT5000-TE``` (2500 test images in VT5000).  
 ### VSOD SOD
 1 train set: ```VSOD-TR```, which is a collection of the train splits in DAVIS and DAVSOD datasets.  
 4 test sets: ```SegV2```, ```FBMS```, ```DAVIS-TE``` and ```DAVSOD-TE```. The last two are the test splits of DAVIS and DAVSOD, respectively.  
 
 
 ## Usage
 Network names used in our framework: Stage 1: ```a2s```; Stage 2: ```cornet``` (RGB SOD), ```midd``` (RGB-D, RGB-T and video SOD).  
 Our train sets include: RGB (DUTS-TR)[c], RGB-D (RGBD-TR)[d], RGB-T (RGB-T)[t] and video (VSOD-TR)[o].
 
 ### Stage 1
 ```
 # RGB SOD
 python3 train.py a2s --gpus=0 --trset=c
 
 # MM-split
 python3 train.py a2s --gpus=0 --trset=[d/o/t]
 
 # MM-joint
 python3 train.py a2s --gpus=0 --trset=cdot
 ```
 
 At the last training epoch in Stage 1, it will generate pseudo labels for all train sets and saves then to a new ```pseudo``` folder.
 
 ### Stage 2
 ```
 python3 train.py a2s --gpus=0 --stage=2 --trset=[c/d/o/t] --vals=[ce/de/oe/te] --weight=path_to_weight [--save]
 # 'ce/de/oe/te' indicates the test sets of RGB, RGB-D, VSOD or RGB-T SOD tasks, respectively.

 ```
 
 ### Test
 ```
 # Stage 1
 python3 test.py a2s --gpus=0 --weight=path_to_weight --vals=[ce/de/oe/te] [--save] [--crf]
 
 # Stage 2
 # RGB SOD
 python3 test.py cornet --gpus=0 --weight=path_to_weight --vals=ce [--save]
 
 # RGB-D, RGB-T or video SOD
 python3 test.py midd --gpus=0 --weight=path_to_weight --vals=[ce/de/oe/te] [--save]
 ```
 

## Results
### RGB SOD  
![Result](https://github.com/moothes/A2S-v2/blob/main/result.PNG)

### Multi-modality SOD  
<div align=center>
<img src="https://github.com/moothes/A2S-v2/blob/main/mm.png">
</div>
