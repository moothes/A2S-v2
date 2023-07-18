# A2S-v2

Source code of our CVPR 2023 paper: "[Texture-guided Saliency Distilling for Unsupervised Salient Object Detection](https://arxiv.org/abs/2207.05921)".    
This work is an improved method of our previous [Activation-to-Saliency (A2S-v1)](https://github.com/moothes/A2S-USOD) published in [TCSVT 2023](https://ieeexplore.ieee.org/document/9875351).  
These two works are based on [SOD benchmark](https://github.com/moothes/SALOD). 

## Resource
You can download the pre-trained [MoCo-v2 weight](https://github.com/facebookresearch/moco) and all trained [weights](https://drive.google.com/drive/folders/1noB7bVjqJqFAYQubTLU_tyF6GkgLZT7z?usp=sharing) of our method.  
RGB SOD results: [pseudo labels](https://drive.google.com/drive/folders/1agLC1iNoONw008jaqEvfRalrBWFWIAL4?usp=sharing) and [saliency maps](https://drive.google.com/drive/folders/15YOcPQ5vzBqlk50DEEVuBXSo5YzqpT89?usp=sharing).  
Results on other multimodal SOD datasets can be easily generated using our code.

 ## Training & Testing
 ### Dataset
 For all SOD tasks, we use the prevalent training sets and re-organized these datasets for convenience.   
 Task | Stage 1 network | Stage 2 network | Training sets | Test sets 
--- | --- | --- | --- | ---
[RGB](https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view?usp=sharing) | ```a2s``` | ```cornet``` | **[cr]** ```DUTS-TR``` or ```MSB-TR``` (the train split of MSRA-B) | **[ce]** ```HKU-IS```, ```PASCAL-S```, ```ECSSD```, ```DUTS-TE```, ```DUT-OMRON```,  ```MSB-TE``` (the test split of MSRA-B) 
[RGB-D](https://drive.google.com/file/d/1mvlkHBqpDal3Ce_gxqZWLzBg4QVWY64U/view?usp=sharing) | ```a2s``` | ```midnet``` | **[dr]** ```RGBD-TR``` or ```RGBD-TR-2985``` | **[de]** ```DUT```, ```LFSD```, ```NJUD```, ```NLPR```, ```RGBD135```, ```SIP```, ```SSD```, ```STERE1000```, ```STEREO```
[RGB-T](https://drive.google.com/file/d/1W-jp9dzUJbWrF6PphKeVk8sLOUiuKT56/view?usp=sharing) | ```a2s``` | ```midnet``` | **[tr]** ```VT5000-TR``` (the train split of VT5000) | **[te]** ```VT821```, ```VT1000``` and ```VT5000-TE``` (the test split of VT5000)
[Video](https://drive.google.com/file/d/1xDvoFflPdlhxR1WSEyrT3dBQLjWADujR/view?usp=sharing) | ```a2s``` | ```midnet``` | **[or]** ```VSOD-TR``` (a collection of the train splits in the DAVIS and DAVSOD datasets) | **[oe]** ```SegV2```, ```FBMS```, ```DAVIS-TE```, ```DAVSOD-TE```
 
Networks ```a2s``` and ```cornet``` are inherited from our previous [A2S-v1](https://github.com/moothes/A2S-USOD) and ```midnet``` is from [here](https://github.com/lz118/Multi-interactive-Dual-decoder).   
```MSB-TR``` and  ```MSB-TE``` are the train+val and test splits of the MSRA-B dataset.   
```RGBD-TR``` (2185 samples, default) and ```RGBD-TR-2985``` (2985 samples) are two different training sets for RGB-D SOD task.  
```VT5000-TR``` and ```VT5000-TE``` are the train and test splits of the VT5000 dataset.   
```VSOD-TR``` is the collection of the train splits of the DAVIS and DAVSOD datasets.
 
 ### Notice
 ```--vals``` has two characters that define the datasets used for testing.   
 **First character** (task): RGB[**c**], RGB-D[**d**], RGB-T[**t**], and video[**o**];   
 **Second character** (phase): training[**r**] or test[**e**] sets.   
 ```--trset``` defines the training sets of different tasks, the same as the first character of ```--vals```.  
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
@inproceedings{zhou2023texture,
  title={Texture-Guided Saliency Distilling for Unsupervised Salient Object Detection},
  author={Zhou, Huajun and Qiao, Bo and Yang, Lingxiao and Lai, Jianhuang and Xie, Xiaohua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7257--7267},
  year={2023}
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



