import tensorflow as tf
import numpy as np
import argparse

from imageProcessing import resizeImages, preprocessImages

from PIL import Image

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def load_labels(label_file):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
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

    image = Image.open(file_name)
    image = np.array(image)

    images = [image, ]
    images = resizeImages(images, (input_width, input_height))
    images = preprocessImages(images)
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