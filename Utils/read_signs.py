import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path
from PIL import Image
import numpy

_GERMAN_DATASET_NUMBER_OF_CLASSES = 43
_BELGIAN_DATASET_NUMBER_OF_CLASSES = 62  # + 10 artificial ones
_ITALIAN_DATASET_NUMBER_OF_CLASSES = 59
_CHINESE_DATASET_NUMBER_OF_CLASSES = 58
_CZECH_DATASET_NUMBER_OF_CLASSES = 17
_FINAL_DATASET_NUMBER_OF_CLASSES = 93


# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
# taken from: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads
def read_traffic_signs(rootpath, classes):
    """Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels"""
    images = []  # images
    labels = []  # corresponding labels
    # loop over all classes
    for c in range(classes):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        with open(prefix + 'GT-' + format(c, '05d') + '.csv') as gtFile:  # annotations file
            gt_reader = csv.reader(gtFile, delimiter=';', )  # csv parser for annotations file
            gt_reader.__next__()  # skip header
            # loop over all images in current annotations file
            for row in gt_reader:
                images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
                labels.append(row[7])  # the 8th column is the label
    return images, labels


def read_german_dataset(path):
    return read_traffic_signs(path, _GERMAN_DATASET_NUMBER_OF_CLASSES)


def read_belgian_dataset(path):
    return read_traffic_signs(path, _BELGIAN_DATASET_NUMBER_OF_CLASSES)


def read_italian_dataset(path):
    images = []
    labels = []

    for c in range(_ITALIAN_DATASET_NUMBER_OF_CLASSES):
        i = 0
        prefix = path + '/' + str(c) + '/track'
        while (Path(prefix + str(i))).is_dir():
            for f in os.listdir(prefix + str(i)):
                img = Image.open(prefix + str(i) + '/' + f)
                images.append(numpy.array(img))
                labels.append(str(c))

            i += 1

    return images, labels


def read_chinese_dataset(path):
    images = []
    labels = []

    for f in os.listdir(path):
        img = Image.open(path + '/' + f)
        images.append(numpy.array(img))
        label = int(f[:3])
        labels.append(str(label))

    return images, labels


def read_czech_dataset(path):
    images = []
    labels = []

    for c in range(_CZECH_DATASET_NUMBER_OF_CLASSES):
        dir_name = path + '/' + str(c)
        for f in os.listdir(dir_name):
            img = Image.open(dir_name + '/' + f)
            images.append(numpy.array(img))
            labels.append(str(c))

    return images, labels


_german_to_final = {'0': 17, '1': 19, '2': 21, '3': 22, '4': 23, '5': 24, '6': 28, '7': 26, '8': 27, '9': 29, '10': 31,
                    '11': 0, '12': 1, '13': 3, '14': 4, '15': 7, '16': 10, '17': 8, '18': 62, '19': 42, '20': 41,
                    '21': 44, '22': 50, '23': 52, '24': 49, '25': 60, '26': 54, '27': 56, '28': 57, '30': 63, '31': 59,
                    '32': 37, '33': 68, '34': 69, '35': 67, '36': 70, '37': 71, '38': 75, '39': 76, '40': 66, '41': 30,
                    '42': 32}

# 32 -> 32 - 70km, 62 - 10km, 63 - 20km, 64 - 25km, 65 - 40km, 66 - 50km, 67 - 30km, 68 - 90km
# (some have km under the number)
# 35 -> 35 - left, 69 - right, 70 - obstacle left, 71 - obstacle right
# Parking signs merged to one category
# 40, 41, 54 look somewhat different
_belgian_to_final = {'0': 50, '1': 51, '2': 52, '3': 42, '4': 41, '5': 44, '6': 43, '8': 61, '9': 58, '10': 60,
                     '12': 65, '11': 54, '13': 62, '14': 47, '15': 48, '16': 49, '17': 0, '18': 45, '19': 3, '20': 5,
                     '21': 4, '22': 8, '23': 11, '25': 10, '28': 7, '29': 35, '30': 34, '31': 29, '32': 23, '33': 79,
                     '34': 67, '35': 74, '36': 70, '37': 66, '38': 78, '40': 38, '41': 39, '44': 6, '45': 85, '46': 85,
                     '47': 85, '48': 85, '49': 85, '50': 85, '51': 90, '52': 91, '53': 80, '54': 82, '56': 81, '59': 92,
                     '60': 2, '61': 1, '62': 15, '63': 17, '64': 18, '65': 20, '66': 21, '67': 19, '68': 25, '69': 73,
                     '70': 76, '71': 75}

# 9 and 23 merged
# 54 looks different
_italian_to_final = {'0': 59, '1': 64, '2': 47, '4': 50, '5': 10, '6': 82, '7': 41, '8': 58, '9': 0, '10': 71, '11': 67,
                     '12': 77, '13': 38, '14': 44, '15': 5, '16': 6, '17': 53, '18': 37, '19': 62, '20': 45, '21': 3,
                     '23': 0, '25': 8, '26': 29, '27': 39, '28': 80, '29': 85, '30': 79, '31': 55, '32': 2, '33': 1,
                     '34': 66, '35': 46, '37': 51, '38': 52, '39': 15, '40': 19, '41': 20, '42': 21, '43': 22, '44': 28,
                     '45': 4, '46': 54, '47': 33, '48': 60, '49': 81, '50': 7, '53': 40, '54': 12, '55': 11, '57': 23,
                     '58': 25}

# 16 is different
_chinese_to_final = {'0': 14, '1': 16, '2': 19, '3': 20, '4': 21, '5': 22, '6': 23, '7': 24, '11': 35, '13': 34,
                     '15': 36, '16': 9, '17': 33, '18': 28, '19': 28, '20': 70, '21': 67, '22': 69, '23': 72, '24': 68,
                     '25': 76, '26': 75, '27': 66, '30': 78, '33': 54, '53': 7, '54': 38, '55': 8}

# some images are unnecessarily big
_czech_to_final = {'0': 38, '1': 81, '2': 82, '3': 1, '4': 39, '5': 83, '6': 84, '7': 6, '8': 13, '9': 2, '10': 8,
                   '11': 7, '12': 79, '13': 86, '14': 87, '15': 88, '16': 89}


# final classes with very few signs: 12, 18, 64, 65, 72, 87, 88, 89, 90, 91
# also 48, 83 and 84 but these can be augmented from other classes


def convert_to_final(final, images, labels, conversion):
    for image, label in zip(images, labels):
        if label in conversion:
            final[conversion[label]].append(image)


def load_and_merge_all_datasets(path):
    final = [[] for _ in range(_FINAL_DATASET_NUMBER_OF_CLASSES)]

    print('Loading german dataset...')
    images, labels = read_german_dataset(path + '/german/data/Training')
    convert_to_final(final, images, labels, _german_to_final)
    images, labels = read_german_dataset(path + '/german/data/Testing')
    convert_to_final(final, images, labels, _german_to_final)
    print('German dataset loaded and converted')

    print('Loading belgian dataset...')
    images, labels = read_belgian_dataset(path + '/belgian/data/Training')
    convert_to_final(final, images, labels, _belgian_to_final)
    images, labels = read_belgian_dataset(path + '/belgian/data/Testing')
    convert_to_final(final, images, labels, _belgian_to_final)
    print('Belgian dataset loaded and converted')

    print('Loading italian dataset...')
    images, labels = read_italian_dataset(path + '/italian/data/Training')
    convert_to_final(final, images, labels, _italian_to_final)
    images, labels = read_italian_dataset(path + '/italian/data/Testing')
    convert_to_final(final, images, labels, _italian_to_final)
    print('Italian dataset loaded and converted')

    print('Loading chinese dataset...')
    images, labels = read_chinese_dataset(path + '/chinese/data/Training')
    convert_to_final(final, images, labels, _chinese_to_final)
    images, labels = read_chinese_dataset(path + '/chinese/data/Testing')
    convert_to_final(final, images, labels, _chinese_to_final)
    print('Chinese dataset loaded and converted')

    print('Loading czech dataset...')
    images, labels = read_czech_dataset(path + '/czech/data')
    convert_to_final(final, images, labels, _czech_to_final)
    print('Czech dataset loaded and converted')

    return final


def make_final_dir(final, path):
    out_dir = path + '/data'
    os.makedirs(out_dir)

    for i in range(_FINAL_DATASET_NUMBER_OF_CLASSES):
        in_dir = out_dir + '/' + format(i, '05d')
        print(f'Creating dir {in_dir} [{len(final[i])} images]')
        os.makedirs(in_dir)

        for j in range(len(final[i])):
            image = final[i][j]
            Image.fromarray(image).save(in_dir + '/' + format(j, '05d') + '.jpg', 'JPEG')


def pipeline(path):
    final = load_and_merge_all_datasets(path)
    make_final_dir(final, path)
    print('*DONE*')


def read_dataset(path):
    images = []
    labels = []

    for c in range(_FINAL_DATASET_NUMBER_OF_CLASSES):
        dir_name = path + '/' + format(c, '05d')
        for f in os.listdir(dir_name):
            img = Image.open(dir_name + '/' + f)
            images.append(numpy.array(img))
            labels.append(str(c))

    return images, labels
