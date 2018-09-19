# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian
#
# taken from: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads

import matplotlib.pyplot as plt
import csv


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def read_traffic_signs(rootpath):
    """Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels"""
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        with open(prefix + 'GT-' + format(c, '05d') + '.csv') as gtFile:  # annotations file
            gt_reader = csv.reader(gtFile, delimiter=';', )  # csv parser for annotations file
            gt_reader.__next__()  # skip header
            # loop over all images in current annotations file
            for row in gt_reader:
                images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
                labels.append(row[7])  # the 8th column is the label
    return images, labels
