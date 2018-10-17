# Utils
import argparse
import inspect
import os
import zipfile
from os.path import join
import requests
import shutil
import sys

import numpy as np
import scipy
# Torch
import torch
import torchvision
from PIL import Image

# DeepDIVA
from util.data.dataset_splitter import split_dataset


def mnist(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the MNIST dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.MNIST(root=args.output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(join(args.output_folder,
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(join(args.output_folder,
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = join(args.output_folder, 'MNIST')
    train_folder = join(dataset_root, 'train')
    test_folder = join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(join(args.output_folder, 'raw'))
    shutil.rmtree(join(args.output_folder, 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def svhn(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the SVHN dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    torchvision.datasets.SVHN(root=args.output_folder, split='train', download=True)
    torchvision.datasets.SVHN(root=args.output_folder, split='test', download=True)

    # Load the data into memory
    train = scipy.io.loadmat(join(args.output_folder,
                                          'train_32x32.mat'))
    train_data, train_labels = train['X'], train['y'].astype(np.int64).squeeze()
    np.place(train_labels, train_labels == 10, 0)
    train_data = np.transpose(train_data, (3, 0, 1, 2))

    test = scipy.io.loadmat(join(args.output_folder,
                                         'test_32x32.mat'))
    test_data, test_labels = test['X'], test['y'].astype(np.int64).squeeze()
    np.place(test_labels, test_labels == 10, 0)
    test_data = np.transpose(test_data, (3, 0, 1, 2))

    # Make output folders
    dataset_root = join(args.output_folder, 'SVHN')
    train_folder = join(dataset_root, 'train')
    test_folder = join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img).save(join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(join(args.output_folder, 'train_32x32.mat'))
    os.remove(join(args.output_folder, 'test_32x32.mat'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def cifar10(args):
    """
    Fetches and prepares (in a DeepDIVA friendly format) the CIFAR dataset to the location specified
    on the file system

    Parameters
    ----------
    args : dict
        List of arguments necessary to run this routine. In particular its necessary to provide
        output_folder as String containing the path where the dataset will be downloaded

    Returns
    -------
        None
    """
    # Use torchvision to download the dataset
    cifar_train = torchvision.datasets.CIFAR10(root=args.output_folder, train=True, download=True)
    cifar_test = torchvision.datasets.CIFAR10(root=args.output_folder, train=False, download=True)

    # Load the data into memory
    train_data, train_labels = cifar_train.train_data, cifar_train.train_labels

    test_data, test_labels = cifar_test.test_data, cifar_test.test_labels

    # Make output folders
    dataset_root = join(args.output_folder, 'CIFAR10')
    train_folder = join(dataset_root, 'train')
    test_folder = join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img).save(join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    os.remove(join(args.output_folder, 'cifar-10-python.tar.gz'))
    shutil.rmtree(join(args.output_folder, 'cifar-10-batches-py'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)


def gw(args):
    url = 'http://www.fki.inf.unibe.ch/DBs/iamHistDB/data/washingtondb-v1.0.zip'
    response = requests.get(url, auth=("unifr", "Zqe2CCyW5ngjshi"))
    os.mkdir("temp")
    output_folder = open(join("temp", "washingtondb-v1.0.zip"), 'wb')
    output_folder.write(response.content)
    gw_zip = zipfile.ZipFile(join("temp", "washingtondb-v1.0.zip"))
    gw_zip.extractall("temp")

    # make split based on spiliting parameter
    base_path = join(args.output_folder, "gw")
    spilt_number = args.gw_split

    # generate base folder and train val test folder
    _make_folder_if_not_exists(base_path)
    _make_folder_if_not_exists(join(base_path, "train"))
    _make_folder_if_not_exists(join(base_path, "test"))
    _make_folder_if_not_exists(join(base_path, "val"))

    word_img_path = join("temp", "washingtondb-v1.0", "data", "word_images_normalized")
    ground_truth_path = join("temp", "washingtondb-v1.0", "ground_truth", "word_labels.txt")
    split_set_path = join("temp", "washingtondb-v1.0", "sets", "cv" + str(spilt_number))

    word_imgs = [f for f in os.listdir(word_img_path) if
                 os.path.isfile(os.path.join(word_img_path, f)) and os.path.splitext(f)[1] == '.png']
    ground_truth = [f for f in open(ground_truth_path, 'r')]

    # get the sets from the split
    training_set = [f[:-1] for f in open(join(split_set_path, "train.txt"))]
    validation_set = [f[:-1] for f in open(join(split_set_path, "valid.txt")).readlines()]
    test_set = [f[:-1] for f in open(join(split_set_path, "test.txt")).readlines()]

    delete_info = ["-", "_pw", "_sq", "_mi", "_qo", "_cm", "_pt", "s_"]

    # iterate over word files

    for k, word_img in enumerate(sorted(word_imgs)):
        word_img_without_extension = os.path.splitext(word_img)[0]
        word_ground_truth = ground_truth[k].replace(word_img_without_extension, "")
        for inf in delete_info:
            word_ground_truth = word_ground_truth.replace(inf, "")

        # trim whitespaces
        word_ground_truth = word_ground_truth.strip()

        # split by train val test set
        text_line = word_img_without_extension[:-3]

        if text_line in training_set:
            # is this word alreadzy existing
            if not os.path.exists(join(base_path, "train", word_ground_truth)):
                os.mkdir(join(base_path, "train", word_ground_truth))
            os.rename(join(word_img_path, word_img), join(base_path, "train", word_ground_truth, word_img))
        if text_line in test_set:
            # is this word alreadzy existing
            if not os.path.exists(join(base_path, "test", word_ground_truth)):
                os.mkdir(join(base_path, "test", word_ground_truth))
            os.rename(join(word_img_path, word_img), join(base_path, "test", word_ground_truth, word_img))
        if text_line in validation_set:
            # is this word alreadzy existing
            if not os.path.exists(join(base_path, "val", word_ground_truth)):
                os.mkdir(join(base_path, "val", word_ground_truth))
            os.rename(join(word_img_path, word_img), join(base_path, "val", word_ground_truth, word_img))

    # TODO delete temp folder
    shutil.rmtree("temp")


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('--dataset',
                        help='name of the dataset',
                        type=str,
                        choices=downloadable_datasets)
    parser.add_argument('--output-folder',
                        help='path to where the dataset should be generated.',
                        required=False,
                        type=str,
                        default='./data/')
    parser.add_argument('--gw-split',
                        help='The split you want to use of the GW dataset for your experiment',
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=1)
    args = parser.parse_args()

    getattr(sys.modules[__name__], args.dataset)(args)
