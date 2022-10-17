# Seeing Beyond the Brain: Masked Modeling Conditioned Diffusion Model for Human Vision Decoding

Author 1, Author 2, ..., Author N

[Paper], [Bibtex], [Supplementary], [Code](https://github.com/zjc062/mind-vis)

## Overview
![Decoding Seen Images from Brain Activities.](/home/zijiao/Desktop/Zijiao/mind-vis-project/mind-vis/figures/first_figure.png)

## Abstract

Decoding visual stimuli from brain recordings aims to deepen our understanding of the human visual system and build a solid foundation for bridging human and computer vision through the Brain-Computer Interface. 
However, due to the scarcity of data annotations and the complexity of underlying brain information, it is challenging to decode images with faithful details and meaningful semantics. 
In this work, we present **MinD-Vis**: Sparse **M**asked Bra**in** Modeling with Double-Conditioned Latent **D**iffusion Model for Human **Vis**ion Decoding.
Specifically, by boosting the information capacity of feature representations learned from a large-scale resting-state fMRI dataset, 
we show that our MinD-Vis can reconstruct highly plausible images with semantically matching details from brain recordings with very few paired annotations.
We benchmarked our model qualitatively and quantitatively; the experimental results indicate that our method outperformed state-of-the-art in both semantic mapping (100-way semantic classification) and generation quality (FID) by **66%** and **41%** respectively. 
Exhaustive ablation studies are conducted to analyze our framework. 


## MinD-Vis Framework
