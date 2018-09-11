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