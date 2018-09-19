This directory contains models created in this project. They are saved in two different formats:
- **.pb** is a Tensorflow low-level format
- **.h5** is a Keras high-level format

Scripts in this repo work mostly with the .pb files but the Keras format can be easier to work with.

All models have been trained on the [German Traffic Sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

## Models
### traffic
This is a convolutional network with two convolution + max-pooling layers and two fully-connected layers. It takes 32x32 grayscale images. It has been trained on the dataset for 5 epochs and reached over **99%** accuracy on the testing dataset.
### traffic_simple
This is a simple network with three fully-connected layers. It also takes 32x32 grayscale images and has been trained for 5 epochs. It reached about **94%** accuracy.
### retrained_MobileNetV2
This model was taken as a module from [Tensorflow Hub](https://alpha.tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/2). It was originally trained on [ImageNet](http://image-net.org/). It takes 96x96 rgb images. It was retrained on the traffic signs dataset for 8000 steps and it reached about **89%** accuracy.

The accuracy is relatively low probably due to the ImageNet dataset having a lot of very different classes whereas traffic signs are relatively similar. The retraining time could also be extended.
