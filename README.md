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

/data
┣ 📂 HCP
┃	┣ 📂 npz
┃   ┃	┣ 📂 dummy_sub_01
┃   ┃ 	┃	┗ HCP_visual_voxel.npz
┃   ┃	┣ 📂 dummy_sub_02
┃   ┃ 	┃	┗ ...

┣ 📂 Kamitani
┃	┣ 📂 npz
┃   ┃	┗ 📜 sbj_1.npz
┃   ┃	┗ 📜 sbj_2.npz
┃   ┃	┗ 📜 sbj_3.npz
┃   ┃	┗ 📜 sbj_4.npz
┃   ┃	┗ 📜 sbj_5.npz
┃   ┃	┗ 📜 images_500.npz
┃   ┃	┗ 📜 imagenet_class_index.json
┃   ┃	┗ 📜 imagenet_training_label.csv
┃   ┃	┗ 📜 imagenet_testing_label.csv

┣ 📂 BOLD5000
┃	┣ 📂 BOLD5000_GLMsingle_ROI_betas
┃ 	┃	┣ 📂 py
┃ 	┃	┃   ┗ CSI1_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_LHEarlyVis.npy
┃ 	┃	┃   ┗ ...
┃ 	┃	┃   ┗ CSIx_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_xx.npy
┃	┣ 📂 BOLD5000_Stimuli
┃ 	┃	┣ 📂 Image_Labels
┃ 	┃	┃   ┗ coco_final_annotations.pkl
┃ 	┃	┃   ┗ imagenet_final_labels.txt
┃ 	┃	┃   ┗ scene_final_labels.txt
┃ 	┃	┣ 📂 Scene_Stimuli
┃   ┃ 	┃	┣ 📂 Original_Images
┃   ┃ 	┃	┣ 📂 Presented_Stimuli
┃   ┃ 	┃	┗ 📜 repeated_stimuli_113_list.txt
┃ 	┃	┣ 📂 Stimuli_Presentation_Lists
┃   ┃ 	┃	┣ 📂 CS1
┃   ┃   ┃ 	┃	┣ 📂 CS1_sess01
┃   ┃   ┃   ┃ 	┃	┗ 📜 CSI_sess01_run01.mat
┃   ┃   ┃   ┃ 	┃	┗ 📜 ...
┃   ┃   ┃ 	┃	┣ 📂 CS1_sess02
┃   ┃   ┃   ┃ 	┃	┗ 📜 CSI_sess02_run01.mat
┃   ┃   ┃   ┃ 	┃	┗ 📜 ...
┃   ┃   ┃ 	┃	┣ 📂 CS1_sess0x
┃   ┃ 	┃	┣ 📂 CS2
┃   ┃   ┃ 	┃	┣ 📂 CS2_sess0x
┃   ┃   ┃ 	┃	┃	┣ (same as CS1_sess0x)
┃   ┃ 	┃	┣ 📂 CS3
┃   ┃   ┃ 	┃	┣ 📂 CS3_sess0x
┃   ┃   ┃ 	┃	┃	┣ (same as CS1_sess0x)
┃   ┃ 	┃	┣ 📂 CS4
┃   ┃   ┃ 	┃	┣ 📂 CS4_sess0x
┃   ┃   ┃ 	┃	┃	┣ (same as CS1_sess0x)


/pretrains
┣ 📂 ldm
┃	┣ 📂 label2img
┃ 	┃	┗ 📜 config.yaml
┃ 	┃	┗ 📜 model.ckpt
┃	┣ 📂 text2img-large
┃ 	┃	┗ 📜 ...
┃	┣ 📂 layout2img
┃ 	┃	┗ 📜 ...
┃	┣ 📂 semantic
┃ 	┃	┗ 📜 ...

┣ 📂 GOD
┃	┗ 📜 fmri_encoder.pth
┃	┗ 📜 finetuned.pth

┣ 📂 BOLD5000
┃	┗ 📜 fmri_encoder.pth
┃	┗ 📜 finetuned.pth


/code
┣ 📂 sc_mbm
┃	┗ 📜 mae_for_fmri.py
┃	┗ 📜 trainer.py
┃	┗ 📜 utils.py

┣ 📂 dc_ldm
┃	┗ 📜 ldm_for_fmri.py
┃	┗ 📜 utils.py
┃	┣ 📂 models
┃	┃	┗ (adopted from latent LDM)
┃	┣ 📂 modules
┃	┃	┗ (adopted from latent LDM)

┗  📜 stageA1_mbm_pretrain.py (main script for pre-training for SC-MBM)
┗  📜 stageA2_mbm_finetune.py (main script for fine-tuning SC-MBM)
┗  📜 stageB_ldm_finetune.py (main script for fine-tuning DC-LDM)
┗  📜 gen_eval.py (main script for evaluating model performance)

┗  📜 dataset.py (functions for loading datasets)
┗  📜 eval_metrics.py (functions for evaluation metrics)
┗  📜 config.py (configurations for the main scripts)

```



## Setup Instructions
### Environment setup
Create and activate conda environment named ```mind_vis``` from our ```env.yaml```
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


