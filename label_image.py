# Run this script from the console to label individual images.
#
# Example usage:
#    python3 label_image.py
#        - to label the example_sign.jpg
#
#    python3 label_image.py --image=path/to/image
#        - to label a different image
#
#    python3 label_image.py --graph=Models/traffic_simple.pb --input_layer=flatten_input --output_layer=dense_2/Softmax
#        - to use the simple model
#
#    python3 label_image.py --graph=Models/retrained_MobileNetV2.pb --input_layer=module_apply_default/hub_input/Mul --output_layer=final_result --input_height=96 --input_width=96 --grayscale=False
#        - to use the retrained MobileNetV2
#
#    python3 label_image.py --help
#        - for more information
#
#
# By default this converts the image to grayscale so make sure to use the --grayscale=False flag if that's not what you want
#
# ==============================================================================


import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

from imageProcessing import resize_images, preprocess_images


def load_graph(model_file):
    """ Load the .pb file into a Tensorflow graph
    :param model_file: location of the .pb file to load
    :return: loaded graph
    """
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    """ Load labels into a list
    :param label_file: location of the labels file to load
    :return: labels
    """
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    file_name = "example_sign.jpg"
    model_file = "Models/traffic.pb"
    label_file = "image_labels.txt"
    input_height = 32
    input_width = 32
    input_layer = "conv2d_input"
    output_layer = "dense_1/Softmax"
    grayscale = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--grayscale", help="flag to convert to grayscale")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    if args.grayscale:
        grayscale = args.grayscale

    image = Image.open(file_name)
    image = np.array(image)

    images = [image, ]
    images = resize_images(images, (input_width, input_height))
    images = preprocess_images(images, grayscale)
    if grayscale == True:
        images = np.expand_dims(images, 4)

    graph = load_graph(model_file)

    input = 'import/' + input_layer
    output = 'import/' + output_layer

    input_operation = graph.get_operation_by_name(input)
    output_operation = graph.get_operation_by_name(output)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], feed_dict={input_operation.outputs[0]: images})

    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i], results[i])
