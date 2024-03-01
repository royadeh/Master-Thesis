# Unsupervised Person ReID

Inspired by other works in unsupervised pseudo labels, we designed a two-stage system for person ReID tasks across different cameras. The task of matching and recognizing persons across multiple camera views or datasets without labeled training data is called unsupervised person re-identification (ReID). Unsupervised person ReID, in contrast to supervised person ReID, which uses annotated data with identification labels for training, tries to develop effective representations exclusively from unannotated data.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

## Introduction
Unsupervised person ReID addresses the challenge of matching individuals across camera viewpoints by learning robust and invariant feature representations.

## Installation
- Install Anaconda and create an environment
- Activate your environment
- Install packages written in requirenment.txt

## Usage
1. Prepare dataset<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Download Market1501and  DukeMTMC-ReID from a website and put the zip file under the directory like

 - `./data`
    - `dukemtmc`
        - `raw`
            - `DukeMTMC-reID.zip`
    - `market1501`
        - `raw`
            - `Market-1501-v15.09.15.zip`

2. Train Model<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You need to  run the train_market.sh in scripts folder
3. Download trained model<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You can download pre-trained weight of Market1501 from [Pretrained_checkpoint_Market1501](https://drive.google.com/file/d/1uTxz8_ozIM7qbL3p3As1upmqJ1jctWXA/view?usp=drive_link)
4. Evaluate Model<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;change the checkpoint path in the test_market.sh in scripts folder and set the trained model.

## Contact
If you have any questions about this code or paper, feel free to contact me at [royadehghani1@gmail.com](royadehghani1@gmail.com)

## Acknowledgement
Codes are built upon [Open-reid](https://github.com/Cysu/open-reid)

# ConvNeXt

ConvNeXt is a CNN-based network designed following the principles of transformers. It is built upon the standard ResNet architecture and incorporates the design features of vision transformers. The key concepts and components that define ConvNeXt include:

## Macro Design and Stages

ConvNeXt is organized into 4 stages, with the number of blocks in each stage set to [3, 3, 9, 3], following the stage compute ratio in Swin transformer (Swin-T) of [1:1:3:1]. The network's macro design also incorporates the "patchify" strategy found in vision transformers. The stem cell in ConvNeXt performs downsampling and includes a $(4 \times 4)$ kernel with a stride of 4.

## ResNeXtify and Inverted Bottleneck

ConvNeXt employs depth-wise and point-wise $(1 \times 1)$ convolutions to separate spatial and channel mixing, similar to the concept of depthwise convolution and weighted sum operation in the self-attention of transformers. The inverted bottleneck architecture, where the MLP (Multi-Layer Perceptron) block's hidden dimension is four times larger than its input dimension, is also integrated into ConvNeXt.

## Depth-wise Convolution and Global Receptive Field

ConvNeXt features a depth-wise kernel size of $(7 \times 7)$ to achieve a global receptive field, akin to the non-local self-attention characteristic of vision transformers. The depth-wise convolution layer is placed at the start of stage 1, allowing for reduced channel dimensions before applying the $(7 \times 7)$ kernel size.

## Activation and Normalization

The ConvNeXt block includes a single Gelu activation function, which is a smoother variant of ReLU. Similar to transformer blocks, each ConvNeXt block is equipped with a single normalization layer—Layer Normalization (LN).

<!-- Image added below -->
<img src="path/to/convnext_image.jpg" alt="ConvNeXt Architecture" width="500px">
