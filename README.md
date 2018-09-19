# DeepTraffic
An online repository for the school project DeepTraffic.

This repo contains various bits and pieces used during the project:

- **Android** directory contains an Android studio project for an app that is a part of this assignment. Read more about it in it's own [README](https://github.com/harmanea/DeepTraffic/tree/master/Android).
- **Models** directory contains models trained during the project. It also has it's own [README](https://github.com/harmanea/DeepTraffic/tree/master/Models).
- **Utils** directory contains pieces of code not written by the author but used for the project.
- The rest of the repository contains other utilities but most notably the *label_image.py* function.

### How to use label_image.py
Run the script from the console to label individual images.

Example usage:

    python3 label_image.py
    # to label the example_sign.jpg

    python3 label_image.py --image=path/to/image
    # to label a different image

    python3 label_image.py --graph=Models/traffic_simple.pb --input_layer=flatten_input --output_layer=dense_2/Softmax
    # to use the simple model

    python3 label_image.py --graph=Models/retrained_MobileNetV2.pb --input_layer=module_apply_default/hub_input/Mul --output_layer=final_result --input_height=96 --input_width=96 --grayscale=False
    # to use the retrained MobileNetV2

    python3 label_image.py --help
    # for more information


### Prerequisities 
This project uses the following libraries:
- [Tensorflow](www.tensorflow.org)
- [matplotlib](matplotlib.org)

You need to install these in the same Python environment for it to work properly.
