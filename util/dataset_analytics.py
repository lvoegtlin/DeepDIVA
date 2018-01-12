"""
This script perform some analysis on the dataset provided.
In particular computes std and mean (to be used to center your dataset).

Structure of the dataset expected:

Split folders
-------------
'args.dataset-folder' has to point to the parent of the train folder.
Example:

        ~/../../data/svhn

where the dataset_folder contains the train sub-folder as follow:

    args.dataset_folder/train

Classes folders
---------------
The train split should have different classes in a separate folder with the class
name. The file name can be arbitrary (e.g does not have to be 0-* for classes 0 of MNIST).
Example:

    train/dog/whatever.png
    train/dog/you.png
    train/dog/like.png

    train/cat/123.png
    train/cat/nsdf3.png
    train/cat/asd932_.png

@author: Michele Alberti
"""

# Utils
import argparse
import os
import sys

import cv2

# Load an color image in grayscale
img = cv2.imread('messi5.jpg', 0)
# Torch related stuff
import torchvision.datasets as datasets

# DeepDIVA
from init.initializer import *


def compute_mean_std(dataset_folder):
    """
    Parameters
    ----------
    :param dataset_folder: String (path)
        Path to the dataset folder (see above for details)

    :return:
        None
    """

    # Getting the train dir
    traindir = os.path.join(dataset_folder, 'train')

    # Sanity check on the training folder
    if not os.path.isdir(traindir):
        logging.error("Train folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)

    # Load the dataset file names
    train_ds = datasets.ImageFolder(traindir)

    # Extract the actual file names and labels as entries
    fileNames = np.asarray([item[0] for item in train_ds.imgs])

    ###############################################################################
    # Compute online mean
    mean = [0, 0, 0]
    for sample in fileNames:
        # NOTE: channels 0 and 2 are swapped because cv2 opens bgr
        img = cv2.imread(sample)
        mean += np.array([np.mean(img[:, :, 2]), np.mean(img[:, :, 1]), np.mean(img[:, :, 0])]) / 255.0

    # Divide by number of samples in train set
    mean /= fileNames.size

    ###############################################################################
    # Compute online mean and standard deviation
    # (see https://stackoverflow.com/questions/15638612/calculating-mean-and-standard-deviation-of-the-data-which-does-not-fit-in-memory)
    std = [0, 0, 0]
    for sample in fileNames:
        # NOTE: channels 0 and 2 are swapped because cv2 opens bgr
        img = cv2.imread(sample) / 255.0
        M2 = np.square(
            np.array([img[:, :, 2] / 255.0 - mean[0], img[:, :, 1] / 255.0 - mean[1], img[:, :, 0] / 255.0 - mean[2]]))
        std += np.sum(np.sum(M2, axis=1), axis=1) / M2.size

    std = np.sqrt(std / fileNames.size)

    # Display the results on console
    print("Mean: [{}, {}, {}]".format(mean[0], mean[1], mean[2]))
    print("Std: [{}, {}, {}]".format(std[0], std[1], std[2]))


if __name__ == "__main__":
    ###############################################################################
    # Argument Parser

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This script perform some analysis on the dataset provided')

    parser.add_argument('--dataset-folder',
                        help='location of the dataset on the machine e.g root/data',
                        required=True,
                        type=str)

    args = parser.parse_args()

    compute_mean_std(dataset_folder=args.dataset_folder)
