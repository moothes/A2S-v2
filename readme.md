# A2S-v2

Our new work on Unsupervised Salient Object Detection (USOD) task.  
Our code is based on our [SOD benchmark](https://github.com/moothes/SALOD).  

Pretrained backbone: [MoCo-v2](https://github.com/facebookresearch/moco).  
RGB SOD results: [pseudo labels](https://drive.google.com/drive/folders/1agLC1iNoONw008jaqEvfRalrBWFWIAL4?usp=sharing) and [saliency maps](https://drive.google.com/drive/folders/15YOcPQ5vzBqlk50DEEVuBXSo5YzqpT89?usp=sharing).  
More resources are coming soon.


 ## Dataset
 For convenience, we re-organized the datasets for different SOD tasks.  
 ### RGB SOD
 2 train sets: ```DUTS-TR``` and ```MSB-TR``` (the train split of MSRA-B).  
 6 test sets: ```HKU-IS```, ```PASCAL-S```, ```ECSSD```, ```DUTS-TE```, ```DUT-OMRON``` and ```MSB-TE``` (the test split of MSRA-B).  
 ### RGB-D SOD
 2 train sets: ```RGBD-TR``` and ```RGBD-TR-2985```. RGBD-TR include 2185 images, the same as other RGB-D SOD methods.  
 9 test sets: ```DUT```, ```LFSD```, ```NJUD```, ```NLPR```, ```RGBD135```, ```SIP```, ```SSD```, ```STERE1000``` and ```STEREO```.  
 ### RGB-T SOD
 1 train set: ```VT5000-TR``` (the train split of VT5000).  
 3 test sets: ```VT821```, ```VT1000``` and ```VT5000-TE``` (the test split of VT5000).  
 ### VSOD SOD
 1 train set: ```VSOD-TR```, which is a collection of the train splits in the DAVIS and DAVSOD datasets.  
 4 test sets: ```SegV2```, ```FBMS```, ```DAVIS-TE``` and ```DAVSOD-TE```. The last two are the test splits of DAVIS and DAVSOD, respectively.  
 
 
 ## Usage
 Network names used in our framework: Stage 1: ```a2s```; Stage 2: ```cornet``` (RGB SOD), ```midd``` (RGB-D, RGB-T and video SOD).  
 Our train sets include: RGB (DUTS-TR), RGB-D (RGBD-TR), RGB-T (VT5000-TR) and video (VSOD-TR).
 
 # notice
 ```--vals``` is formated as two characters.   
 The first character means the task: (RGB (c), RGB-D (d), RGB-T (t) and video (o).)  
 The second character means train (r) or test (e) sets.  
 For example, *cr* incidates the train sets of the RGB SOD task, *oe* indicates the test sets of the VSOD task.   
 More details please refer to ```data.py```
 
 
 ### Stage 1
 ```
 # RGB SOD
 python3 train.py a2s --gpus=0 --trset=c
 
 # MM-split
 python3 train.py a2s --gpus=0 --trset=[d/o/t]
 
 # MM-joint
 python3 train.py a2s --gpus=0 --trset=cdot
 ```
 
 At the last training epoch in Stage 1, it will generate pseudo labels for all train sets and save them to a new ```pseudo``` folder.
 
 ### Stage 2
 ```
 python3 train.py a2s --gpus=0 --stage=2 --trset=[c/d/o/t] --vals=[ce/de/oe/te] --weight=path_to_weight [--save]
 ```
 
 ### Test
 ```
 # Stage 1
 python3 test.py a2s --gpus=0 --weight=path_to_weight --vals=[ce/de/oe/te] [--save] [--crf]
 
 # Stage 2
 ## RGB SOD
 python3 test.py cornet --gpus=0 --weight=path_to_weight --vals=ce [--save]
 
 ## RGB-D, RGB-T or video SOD
 python3 test.py midd --gpus=0 --weight=path_to_weight --vals=[de/oe/te] [--save]
 ```
 
 

## Results
### RGB SOD  
![Result](https://github.com/moothes/A2S-v2/blob/main/result.PNG)

### Multi-modality SOD  
<div align=center>
<img src="https://github.com/moothes/A2S-v2/blob/main/mm.png">
</div>
