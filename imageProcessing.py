import random

import numpy
from PIL import Image


def resize_images(images, size):
    """ Resize images in numpy arrays to a new specified size using the PIL Image.resize function
    :param images: list of images in numpy arrays to be resized
    :param size: new size for the images
    :return: list of resized images
    """
    resized_images = []
    for image in images:
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize(size, Image.ANTIALIAS)
        resized_images.append(numpy.array(resized_image))
    return resized_images


def preprocess_images(images, grayscale=True):
    """ Convert images to grayscale and scale their values between 0 and 1
    :param images: list of images to be processed
    :param grayscale: flag for converting the images to grayscale
    :return: list of processed images
    """
    processed_images = []
    for image in images:

        # convert to grayscale, YCbCr representation
        if grayscale == True:
            processed_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            processed_image = image

        # scale features between 0 and 1
        processed_image = processed_image / 255.0

        processed_images.append(processed_image)
    return processed_images


# Taken from stackoverflow answer by sshashank124
# https://stackoverflow.com/a/23289591/9943257
def unison_shuffle(a, b):
    """ Shuffles two lists in unison
    :param a: first list
    :param b: second list
    :return: shuffled lists
    """
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    return zip(*c)


# puts all necessary steps together
def process(images, labels, grayscale=True, shuffle=True):
    """ Resize, convert to grayscale, scale their values between 0 and 1 and shuffle images and labels
    :param images: list of images to be processed
    :param labels: list of labels corresponding to the images
    :param grayscale: flag for converting the images to grayscale
    :param shuffle: flag for shuffling lists, True by default
    :return: processed lists
    """
    images = resize_images(images, (32, 32))
    images = preprocess_images(images, grayscale)
    if shuffle:
        images, labels = unison_shuffle(images, labels)
    images = numpy.array(images)
    if grayscale:
        images = numpy.expand_dims(images, 4)
    labels = numpy.array(labels)
    return images, labels


class_names = ['speed limit 20', 'speed limit 30', 'speed limit 50', 'speed limit 60', 'speed limit 70',
               'speed limit 80', 'end of speed limit 80', 'speed limit 100', 'speed limit 120', 'no passing',
               'no passing by heavy vehicles', 'priority road at next intersection', 'priority road', 'yield', 'stop',
               'no vehicles permitted', 'no heavy vehicles permitted', 'do not enter', 'danger', 'left curve',
               'right curve', 'double curves', 'bumpy road', 'slippery road', 'narrow road', 'road workers',
               'semaphore', 'pedestrians', 'children', 'bycicles',
               'snow or ice', 'wild animals', 'end of speed limit', 'right turn', 'left turn', 'go straight',
               'go straight or right turn', 'go straight or left turn', 'keep right', 'keep left',
               'roundabout', 'end of no passing', 'end of no passing by heavy vehicles']
