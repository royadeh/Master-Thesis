# Unsupervised Person ReID
Inspired by other works in unsupervised pseudo Llabels-based methods, we designed a two-satge system for person ReID task across different cameras. The task of matching and recognising persons across multiple camera views or datasets without the use of labelled training data is referred to as unsupervised person re-identification (ReID). We Investigate the use of ConvNeXt into IICS/IIDS framework for person Re-ID.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#Contact)
- [Acknowledgement](#Acknowledgement)


## Introduction
Unsupervised person ReID addresses the challenge of matching individuals across different camera viewpoints by learning robust and invariant feature representations.

## Installation
- Install Anaconda and create an environment
- Activate your environment
- Install packages written in requirenment.txt

## Usage
1. Prepare dataset<br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Download Market1501and  DukeMTMC-ReID from website and put the zip file under the directory like

 - `./data`
    - `dukemtmc`
        - `raw`
            - `DukeMTMC-reID.zip`
    - `market1501`
        - `raw`
            - `Market-1501-v15.09.15.zip`

    - `msmt17`
        - `raw`
            - `MSMT17-V2.09.15.zip`
2. Train Model<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You need to  run the train_market.sh in scripts folder
3. Download pre-trained model<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You can download pretrained weight of Market1501 from [Pretrained_checkpoint_Market1501](https://drive.google.com/file/d/1uTxz8_ozIM7qbL3p3As1upmqJ1jctWXA/view?usp=drive_link)
4. Evaluate Model<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;change the checkpoint path in the test_market.sh in scripts folder and set the trained model.

## Contact
If you have any questions about this code or paper, feel free to contact me at [royadehghani1@gmail.com](royadehghani1@gmail.com)

## Acknowledgement
Codes are built upon [Open-reid](https://github.com/Cysu/open-reid)
