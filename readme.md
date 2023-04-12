# A2S-v2

Source code of our CVPR 2023 paper: "[Texture-guided Saliency Distilling for Unsupervised Salient Object Detection](https://arxiv.org/abs/2207.05921)".    
This work is an improved method of our previous [Activation-to-Saliency (A2S-v1)](https://github.com/moothes/A2S-USOD) published in [TCSVT 2023](https://ieeexplore.ieee.org/document/9875351).  
These two works are based on [SOD benchmark](https://github.com/moothes/SALOD), which provide an easy way for building new SOD methods. 

## Resource
Pretrained backbone: [MoCo-v2](https://github.com/facebookresearch/moco).  
All trained weights can be download from [Onedrive](https://drive.google.com/drive/folders/1noB7bVjqJqFAYQubTLU_tyF6GkgLZT7z?usp=sharing).  
RGB SOD results: [pseudo labels](https://drive.google.com/drive/folders/1agLC1iNoONw008jaqEvfRalrBWFWIAL4?usp=sharing) and [saliency maps](https://drive.google.com/drive/folders/15YOcPQ5vzBqlk50DEEVuBXSo5YzqpT89?usp=sharing).  

 ### Dataset
 For convenience, we re-organized the datasets for different SOD tasks.  
 #### [RGB](https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view?usp=sharing)
 2 training sets: ```DUTS-TR``` or ```MSB-TR``` (the train split of MSRA-B).  
 6 test sets: ```HKU-IS```, ```PASCAL-S```, ```ECSSD```, ```DUTS-TE```, ```DUT-OMRON``` and ```MSB-TE``` (the test split of MSRA-B).  
 #### [RGB-D](https://drive.google.com/file/d/1mvlkHBqpDal3Ce_gxqZWLzBg4QVWY64U/view?usp=sharing)
 2 training sets: ```RGBD-TR``` or ```RGBD-TR-2985```. RGBD-TR include 2185 images, the same as other RGB-D SOD methods.  
 9 test sets: ```DUT```, ```LFSD```, ```NJUD```, ```NLPR```, ```RGBD135```, ```SIP```, ```SSD```, ```STERE1000``` and ```STEREO```.  
 #### [RGB-T](https://drive.google.com/file/d/1W-jp9dzUJbWrF6PphKeVk8sLOUiuKT56/view?usp=sharing)
 1 training set: ```VT5000-TR``` (the train split of VT5000).  
 3 test sets: ```VT821```, ```VT1000``` and ```VT5000-TE``` (the test split of VT5000).  
 #### [Video](https://drive.google.com/file/d/1xDvoFflPdlhxR1WSEyrT3dBQLjWADujR/view?usp=sharing)
 1 training set: ```VSOD-TR``` (a collection of the train splits in the DAVIS and DAVSOD datasets).  
 4 test sets: ```SegV2```, ```FBMS```, ```DAVIS-TE``` and ```DAVSOD-TE```. The last two sets are the test splits of DAVIS and DAVSOD, respectively.  
 
 
 ## Training & Testing
 Network names used in our framework: Stage 1: ```a2s```; Stage 2: ```cornet``` (RGB), ```midnet``` (RGB-D, RGB-T, and video).  
 Our training sets include: ```DUTS-TR```/```MSB-TR```, ```RGBD-TR```, ```VT5000-TR```, and ```VSOD-TR```.
 
 ### Notice
 ```--vals``` is formated as two characters to define the test sets.   
 **First character** (task): RGB[**c**], RGB-D[**d**], RGB-T[**t**], and video[**o**];   
 **Second character** (dataset): training[**r**] or test[**e**] sets.  
 For example, ''cr'' incidates the training sets of RGB SOD task, ''oe'' indicates the test sets of VSOD task.   
 ```--trset``` defines the training sets of different tasks used for training, similar with the first character of ```--vals```.  
 More details please refer to ```data.py```.
 
 ### Stage 1
 ```
 ## Training
 # Training for RGB SOD task
 python3 train.py a2s --gpus=[gpu_num] --trset=c
 
 # Split training for single multimodal task
 python3 train.py a2s --gpus=[gpu_num] --trset=[d/o/t]
 
 # Joint training for four multimodal tasks
 python3 train.py a2s --gpus=[gpu_num] --trset=cdot
 
 ## Testing
 # Generating pseudo labels
 python3 test.py a2s --gpus=[gpu_num] --weight=[path_to_weight] --vals=[cr/dr/or/tr] --save --crf
 
 # Testing on test sets
 python3 test.py a2s --gpus=[gpu_num] --weight=[path_to_weight] --vals=[ce/de/oe/te] [--save]
 ```
 
 After the training process in stage 1, we will generate pseudo labels for all training sets and save them to a new ```pseudo``` folder.
 
 ### Stage 2
 ```
 ## Training
 # Training for RGB SOD task
 python3 train.py cornet --gpus=[gpu_num] --stage=2 --trset=c --vals=ce
 
 # Training for RGB-D, RGB-T or video SOD tasks
 python3 train.py midnet --gpus=[gpu_num] --stage=2 --trset=[d/o/t] --vals=[de/oe/te]
 
 ## Testing
 python3 test.py [cornet/midnet] --gpus=[gpu_num] --weight=[path_to_weight] --vals=[de/oe/te] [--save]
 ```
## Reference 
Thanks for citing our serial works.
```xml
@article{zhou2023a2s2,
  title={Texture-guided Saliency Distilling for Unsupervised Salient Object Detection},
  author={Zhou, Huajun and Qiao, Bo and Yang, Lingxiao and Lai, Jianhuang and Xie, Xiaohua},
  journal={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2023},
  publisher={IEEE}
}

@ARTICLE{zhou2023a2s1,
  title={Activation to Saliency: Forming High-Quality Labels for Unsupervised Salient Object Detection}, 
  author={Zhou, Huajun and Chen, Peijia and Yang, Lingxiao and Xie, Xiaohua and Lai, Jianhuang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2023},
  volume={33},
  number={2},
  pages={743-755},
  doi={10.1109/TCSVT.2022.3203595}}
```

## Results
![Result](https://github.com/moothes/A2S-v2/blob/main/rgb.PNG)
 
<div align=center>
<img src="https://github.com/moothes/A2S-v2/blob/main/mm.PNG", width=900>
</div>



