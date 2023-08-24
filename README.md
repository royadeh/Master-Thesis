# Adaptive Instance and Batch Normalisation (AIBN) and Transform Normalization (TNorm) in Person Re-Identification

## AIBN
The purpose of AIBN (Adaptive Instance and Batch Normalisation) is to address the challenges posed by intra-camera variations in person Re-ID. AIBN combines the benefits of Instance Normalization (IN) and Batch Normalization (BN) to make the network robust against identity-related changes while maintaining individual variance. The AIBN technique blends statistics obtained from both IN and BN in an adaptive manner, enhancing the discriminatory power of the network.

To integrate AIBN into ConvNeXt, the Layer Normalization (LN) in ConvNeXt Blocks is replaced with AIBN in the final stages (Stage 3 and Stage 4).

## TNorm
Transform Normalization (TNorm) is a technique proposed to mitigate the effects of camera-related factors on deep feature representations in person Re-ID. TNorm normalizes the feature vectors of images based on statistics from the same camera, reducing variations caused by lighting conditions, illumination levels, and image resolutions. The feature statistics within a mini-batch of each camera are efficiently computed for TNorm.

TNorm can be used for data augmentation by randomly selecting the target camera. To integrate TNorm into the ConvNext network, the TNorm layers are added after Stage 1, Stage 2, and Stage 3 in the ConvNeXt architecture.


