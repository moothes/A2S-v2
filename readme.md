# A2S-v2

## News: Our work is accepted by CVPR 2023. 
Source code of our work: "[Texture-guided Saliency Distilling for Unsupervised Salient Object Detection](https://arxiv.org/abs/2207.05921)".    
Our code is based on [A2S-v1](https://github.com/moothes/A2S-USOD) and [SOD benchmark](https://github.com/moothes/SALOD).

Pretrained backbone: [MoCo-v2](https://github.com/facebookresearch/moco).  
All trained weights can be download from [Onedrive](https://drive.google.com/drive/folders/1noB7bVjqJqFAYQubTLU_tyF6GkgLZT7z?usp=sharing).  
RGB SOD results: [pseudo labels](https://drive.google.com/drive/folders/1agLC1iNoONw008jaqEvfRalrBWFWIAL4?usp=sharing) and [saliency maps](https://drive.google.com/drive/folders/15YOcPQ5vzBqlk50DEEVuBXSo5YzqpT89?usp=sharing).  
More resources are coming soon.

 ## Dataset
 For convenience, we re-organized the datasets for different SOD tasks.  
 ### RGB SOD [download](https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view?usp=sharing)
 2 training sets: ```DUTS-TR``` or ```MSB-TR``` (the train split of MSRA-B).  
 6 test sets: ```HKU-IS```, ```PASCAL-S```, ```ECSSD```, ```DUTS-TE```, ```DUT-OMRON``` and ```MSB-TE``` (the test split of MSRA-B).  
 ### RGB-D SOD [download](https://drive.google.com/file/d/1mvlkHBqpDal3Ce_gxqZWLzBg4QVWY64U/view?usp=sharing)
 2 training sets: ```RGBD-TR``` or ```RGBD-TR-2985```. RGBD-TR include 2185 images, the same as other RGB-D SOD methods.  
 9 test sets: ```DUT```, ```LFSD```, ```NJUD```, ```NLPR```, ```RGBD135```, ```SIP```, ```SSD```, ```STERE1000``` and ```STEREO```.  
 ### RGB-T SOD [download](https://drive.google.com/file/d/1W-jp9dzUJbWrF6PphKeVk8sLOUiuKT56/view?usp=sharing)
 1 training set: ```VT5000-TR``` (the train split of VT5000).  
 3 test sets: ```VT821```, ```VT1000``` and ```VT5000-TE``` (the test split of VT5000).  
 ### Video SOD [download](https://drive.google.com/file/d/1xDvoFflPdlhxR1WSEyrT3dBQLjWADujR/view?usp=sharing)
 1 training set: ```VSOD-TR``` (a collection of the train splits in the DAVIS and DAVSOD datasets).  
 4 test sets: ```SegV2```, ```FBMS```, ```DAVIS-TE``` and ```DAVSOD-TE```. The last two sets are the test splits of DAVIS and DAVSOD, respectively.  
 
 
 ## Usage
 Network names used in our framework: Stage 1: ```a2s```; Stage 2: ```cornet``` (RGB SOD), ```midd``` (RGB-D, RGB-T and video SOD).  
 Our train sets include: DUTS-TR/MSB-TR, RGBD-TR, VT5000-TR and VSOD-TR.
 
 ### Notice
 ```--vals``` is formated as two characters.   
 The first character means the task: (RGB[c], RGB-D[d], RGB-T[t] and video[o].)  
 The second character means training[r] or test[e] sets.  
 For example, ''cr'' incidates the train sets of the RGB SOD task, ''oe'' indicates the test sets of the VSOD task.   
 ```--trset``` defines the training sets of different tasks used for training, similar with the first character of ```--vals```.  
 More details please refer to ```data.py```.
 
 
 ### Stage 1
 ```
 # RGB SOD
 python3 train.py a2s --gpus=0 --trset=c
 
 # Single multimodal dataset
 python3 train.py a2s --gpus=0 --trset=[d/o/t]
 
 # Joint multimodal dataset
 python3 train.py a2s --gpus=0 --trset=cdot
 ```
 
 After the training process in stage 1, we will generate pseudo labels for all training sets and save them to a new ```pseudo``` folder.
 
 ### Stage 2
 ```
 # RGB
 python3 train.py cornet --gpus=0 --stage=2 --trset=c --vals=ce [--save]
 
 # RGB-D, RGB-T or video SOD
 python3 train.py midd --gpus=0 --stage=2 --trset=[d/o/t] --vals=[de/oe/te] [--save]
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
![Result](https://github.com/moothes/A2S-v2/blob/main/rgb.PNG)
 
<div align=center>
<img src="https://github.com/moothes/A2S-v2/blob/main/mm.PNG", width=900>
</div>



Thanks for citing our work
```xml
@article{zhou2022a2s2,
  title={Texture-guided Saliency Distilling for Unsupervised Salient Object Detection},
  author={Zhou, Huajun and Qiao, Bo and Yang, Lingxiao and Lai, Jianhuang and Xie, Xiaohua},
  journal={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2023},
  publisher={IEEE}
}
```
