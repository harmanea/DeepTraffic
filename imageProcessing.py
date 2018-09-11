from PIL import Image
import numpy

def resizeImages(images, size):
    '''
    Resizes images in numpy arrays to a new specified size using the PIL Image.resize function
    :param images: list of images in numpy arrays to be resized
    :param size: new size for the images
    :return: list of resized images
    '''
    resizedImages = []
    for image in images:
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize(size, Image.ANTIALIAS)
        resizedImages.append(numpy.array(resized_image))
    return resizedImages

def preprocessImages(images):
    '''
    Convert images to grayscale and scale their values between 1 and 2
    :param images: list of images to be processed
    :return: list of processed images
    '''
    processedImages = []
    for image in images:
        # convert to grayscale, YCbCr representation
        processedImage = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

        #scale features between 0 and 1
        processedImage = processedImage / 255.0

        processedImages.append(processedImage)
    return processedImages


import random

# Taken from stackoverflow answer by sshashank124
# https://stackoverflow.com/a/23289591/9943257
def unisonShuffle(a, b):
    '''
    Shuffles two lists in unison
    :param a: first list
    :param b: second list
    :return: shuffled lists
    '''
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    return zip(*c)

# puts all necessary steps together
def process(images, labels, shuffle=True):
    images = resizeImages(images, (32, 32))
    images = preprocessImages(images)
    if shuffle:
        images, labels = unisonShuffle(images, labels)
    images = numpy.array(images)
    images = numpy.expand_dims(images, 4)
    labels = numpy.array(labels)
    return images, labels