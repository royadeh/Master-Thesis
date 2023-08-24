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


