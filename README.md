# Deep-Learning Super-resolution Image Reconstruction (DSIR)

Super-resolution microscopy techniques (PALM, STORM…) can improve enormously localization resolution by acquiring many single emitters emission (fluorescence labels) of sparse events. Low density (LD) data acquisition can provide tens of thousand of images of sparse events that can be readily localized using standard fitting algorithms (e.g, Thunder-STORM). However, the aquisiton of such large number of images takes some time and can result to the sample drift and degrate the sample due to the intense light exposion. High density (HD) data of several hundred images can be faster in acquisition time but result of a very dense number of events per frames, which compromise the performance of the fitting localization algorithms. 

This repository proposes a method that use convolution neural network (ConvNet) Auto-encoder to reconstruct a localization image from HD datasets. 

![convnet autoencoder](https://gitlab.icfo.net/leaxp/deep-learning-super-resolution-image-reconstruction/raw/assets/localization.png)





