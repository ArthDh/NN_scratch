import numpy as np
import os

IMG_HEIGHT = 0
IMG_WIDTH = 0


def convertInt(data, width, height):
    val = []
    for i in range(height):
        for j in range(width):
            if data[i][j] == ' ':
                val.append(0)
            else:
                val.append(1)
    return val


def loadDataFile(filename, n, width, height):
    fin = readlines(filename)
    fin.reverse()
    items = []
    IMG_WIDTH = width
    IMG_HEIGHT = height
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < IMG_WIDTH - 1:
            # we encountered end of file...
            print("Truncating at %d examples (maximum)" % i)
            break
        items.append(convertInt(data, width, height))
    return items


def readlines(filename):
    # "Opens a file or reads it from the zip archive data.zip"
    if(os.path.exists(filename)):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).split('\n')


def loadLabelsFile(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels


def getDataLabels(data_path, labels_path, n_examples, img_shape=(60, 70)):
    # Length of training data
    training_size = n_examples
    # Loading training Data
    X = loadDataFile(data_path, training_size, img_shape[0], img_shape[1])
    # Loading training labels
    Y = loadLabelsFile(labels_path, training_size)
    # Example of training data (84x37)
    # print(X[-1])
    # Labels of specified data point
    # print(Y[-1])
    print("Length of training data:", len(X))
    print("Length of training labels:", len(Y))
    return X, Y
