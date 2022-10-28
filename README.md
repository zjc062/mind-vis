# Seeing Beyond the Brain: Masked Modeling Conditioned Diffusion Model for Human Vision Decoding

## MinD-Vis
**MinD-Vis** is a framework for decoding human visual stimuli from brain recording.
This document introduces the precesedures required for replicating the results in *Seeing Beyond the Brain: Masked Modeling Conditioned Diffusion Model for Human Vision Decoding* (Submitted to CVPR2022)

## Abstract
Decoding visual stimuli from brain recordings aims to deepen our understanding of the human visual system and build a solid foundation for bridging human and computer vision through the Brain-Computer Interface. However, due to the scarcity of data annotations and the complexity of underlying brain information, it is challenging to decode images with faithful details and meaningful semantics. In this work, we present **MinD-Vis**: Sparse **M**asked Bra**in** Modeling with Double-Conditioned Latent **D**iffusion Model for Human **Vis**ion Decoding. Specifically, by boosting the information capacity of feature representations learned from a large-scale resting-state fMRI dataset, we show that our MinD-Vis can reconstruct highly plausible images with semantically matching details from brain recordings with very few paired annotations. We benchmarked our model qualitatively and quantitatively; the experimental results indicate that our method outperformed state-of-the-art in both semantic mapping (100-way semantic classification) and generation quality (FID) by **66%** and **41%** respectively. 


## Overview
Our framework consists of two main stages:
- Stage A: Sparse-Coded Masked Brain Modelling (*SC-MBM*)
- Stage B: Double-Conditioned Latent Diffusion Model (*DC-LDM*)

File path | Description
```
**TO BE UPDATED**
/data
┣ 📂 imagenet
┃	┣ 📂 val
┃ 	┃	┗ (ImageNet validation images by original class folders)

┣ 📂 imagenet_depth
┃	┣ 📂 val_depth_on_orig_small_png_uint8
┃	┃	┗ (depth component of ImageNet validation images using MiDaS small model)
┃	┣ 📂 val_depth_on_orig_large_png_uint8
┃	┃	┗ (depth component of ImageNet validation images using MiDaS large model)

┣ 📂 imagenet_rgbd
┃	┗	(pretrained depth-only & RGBD vgg16/19 model checkpoints optimized for ImageNet classification challenge; These are used as Encoder backbone net or as a reconstruction metric)

┣ 📜 images_112.npz (fMRI on ImageNet stimuli at resolution 112x112)
┣ 📜 rgbd_112_from_224_large_png_uint8.npz (saved RGBD data at resolution 112, depth computed on 224 stimuli using MiDaS large model and saved as PNG uint8)
┣ 📜 sbj_<X>.npz (fMRI data)
┗ 📜 model-<X>.pt (MiDaS depth estimation models)

/code
┣ 📂 sc_mbm
┃	┣ 📂 val
┃ 	┃	┗ (ImageNet validation images by original class folders)

┣ 📂 dc_ldm
┃	┣ 📂 val_depth_on_orig_small_png_uint8
┃	┃	┗ (depth component of ImageNet validation images using MiDaS small model)
┃	┣ 📂 val_depth_on_orig_large_png_uint8
┃	┃	┗ (depth component of ImageNet validation images using MiDaS large model)

┗  📜 stageA1_mbm_pretrain.py (main script for pre-training for SC-MBM)
┗  📜 stageA2_mbm_finetune.py (main script for fine-tuning SC-MBM)
┗  📜 stageB_ldm_finetune.py (main script for fine-tuning DC-LDM)


```



## Setup Instructions
### Environment setup
Create and activate conda environment named ```mind_vis``` from our ```environment.yaml```
```sh
conda env create -f environment.yaml
conda activate mind_vis
```

### Download data, pre-trained model and replicate the whole framework
To replicate the main result (i.e. **Full model**, Stage A + Stage B) shown in our submitted manuscript, please modify the file paths in ```xxx.sh``` and run
```sh  
sh ./mind-vis/scripts/xxxx
```
This script will automatically download the required data and pre-trained models in ```./mind-vis/data```  and ```./mind-vis/pretrains```. Results will be saved in ```./mind-vis/results```. The whole procedure took xxx days on a RTX3090 GPU (Stage A: xxx days; Stage B: xxx hours). 

### Replicate the result for **Stage A: Sparse-Coded Masked Brain Modelling**
Please modify the file paths in ```xxx.sh``` and run
```sh  
sh ./mind-vis/scripts/xxxx
```
This script will automatically download the required data (HCP dataset and KAM dataset) in ```./mind-vis/data```. Results will be saved in ```./mind-vis/results/stageA_fmri_pretrain```. The pre-trained checkpoint for downstream image generation will be saved in ```./mind-vis/pretrains/fmri_pretrain```
This stage took xxx days on a RTX3090 GPU (Stage A1: xxx days; Stage A2: xxx days). 

### Replicate the result for **Stage B: Double-Conditioned Latent Diffusion Model**
Please modify the file paths in ```xxx.sh``` and run
```sh  
sh ./mind-vis/scripts/xxxx
```
This script will automatically download the required pretrained LDM checkpoint in ```./mind-vis/pretrains/ldm```. Results will be saved in ```./mind-vis/results/stageB_image_generation```. The pre-trained checkpoint for downstream image generation will be saved in ```./mind-vis/pretrains/fmri_pretrain```
This stage took xxx hours on a RTX3090 GPU. 



### Replicate the result for ablation studies for Stage A
Please modify the file paths in ```xxx.sh```, update the flag information and run
```sh
sh ./mind-vis/scripts/xxxx
```
e.g. update the flag information
```sh
python xxx.py --flag1 a --flag2 b
```
This script requires the datasets downloaded in ```./mind-vis/data```. Results will be saved in ```./mind-vis/results/stageA_ablation```. The pre-trained checkpoint for downstream image generation will be saved in ```./mind-vis/pretrains/fmri_pretrain```

### Replicate the result for ablation studies for Stage B
Please modify the file paths in ```xxx.sh```, update the flag information and run
```sh
sh ./mind-vis/scripts/xxxx
```
e.g. update the flag information
```sh
python xxx.py --flag1 a --flag2 b
```
This script requires the datasets downloaded in ```./mind-vis/data```; checkpoints trained and saved in ```./mind-vis/pretrains/fmri_pretrain```. Results will be saved in ```./mind-vis/results/stageB_ablation```. The pre-trained checkpoint for downstream image generation will be saved in ```./mind-vis/pretrains/fmri_pretrain```

### Testing, evaluation and visualization
The ```xxx.ipynb``` notebook in ```./mind-vis/scripts``` provides functionality for evaluating reconstruction quality via n-way identification experiments and FID score, and visualizing the reconstructed images.


