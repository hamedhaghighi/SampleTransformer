# SampleTransformer: Modeling high fidelity music in raw audio domain
In this project, we attempt to scale sparse transformer architecture for long-range dependencies. we construct a network including 40 layers of transformer layers plus two wavenet model which is trained on 2 seconds of music but with the help of non-trainable memory, the context for prediciting next sample is extended to 8 seconds.   

## Dependencies

* tensorflow == 1.13
* Librosa
* tqdm
* blocksparse

## Dataset

Music dataset introduced in paper [SampleRNN](https://arxiv.org/abs/1612.07837).

## Overview

Our network consists of two wavenets (at the begining and the ), sparse transformer blocks and a transformer block in the middle. Similar to U-Net architecure, there are down\up sampling modules following each sparse transformer block,  connecting to each other symmetrically. Inspired by [transformer XL](https://arxiv.org/abs/1901.02860), transformer block , in the middle, attends to deatched memory to increase sample context. Architecure of the network is shown in Fig.1.

![arch](images/arch.png)

*Fig.1 architecure of the network. (a) Overall archietecure (b) inside transformer blocks (sparse or normal)*


## Usage:

Install forked [blocksparse](https://github.com/hamedhaghighi/blocksparse) by compiling the source code.  
Run training use command below:  
    python main.py
    
Generation WIP ...