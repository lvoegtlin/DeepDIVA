from itertools import chain

import logging
import sys
import numpy as np
import torch
from skimage import io as img_io

from skimage.transform import resize
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from template.runner.key_word_spotting.homography_augmentation import HomographyAugmentation


def load_dataset(dataset_folder, in_memory=False, workers=1, phoc_unigram_levels=(1, 2, 4, 8)):
    """
        Loads the dataset from file system and provides the dataset splits for train validation and test.

        The dataset is expected to be in the same structure as described in image_folder_dataset.load_dataset()

        Parameters
        ----------
        dataset_folder : string
            Path to the dataset on the file System
        in_memory : boolean
            Load the whole dataset in memory. If False, only file names are stored and images are loaded
            on demand. This is slower than storing everything in memory.
        workers: int
            Number of workers to use for the dataloaders

        Returns
        -------
        train_ds : data.Dataset
        val_ds : data.Dataset
        test_ds : data.Dataset
            Train, validation and test splits
        """
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder, 'val')
    test_dir = os.path.join(dataset_folder, 'test')

    # Sanity check on the splits folders
    if not os.path.isdir(train_dir):
        logging.error("Train folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(val_dir):
        logging.error("Val folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)
    if not os.path.isdir(test_dir):
        logging.error("Test folder not found in the args.dataset_folder=" + dataset_folder)
        sys.exit(-1)

    train_ds = ImageFolderWordSpotting(train_dir, phoc_unigram_levels=phoc_unigram_levels, dataset_type='train')
    val_ds = ImageFolderWordSpotting(val_dir, phoc_unigram_levels=phoc_unigram_levels, dataset_type='val')
    test_ds = ImageFolderWordSpotting(test_dir, phoc_unigram_levels=phoc_unigram_levels, dataset_type='test')

    return train_ds, val_ds, test_ds


"""
Created on Sep 3, 2017

@author: ssudholt
original code from https://github.com/georgeretsi/pytorch-phocnet
"""


class ImageFolderWordSpotting(Dataset):
    '''
    PyTorch dataset class for the segmentation-based George Washington dataset
    '''

    def __init__(self, path, phoc_unigram_levels, dataset_type='train', workers=None):
        '''
        Constructor

        path : string
            Path to the dataset on the file System
        workers: int
            Number of workers to use for the dataloaders
        '''

        # class members
        self.word_string_embeddings = None
        self.query_list = None
        self.label_encoder = None
        self.weights = None
        self.fixed_image_size = None

        # load the dataset
        labels = sorted([elem for elem in os.listdir(path)])

        words = []
        for label in labels:
            for word_img_name in [elem for elem in os.listdir(os.path.join(path, label))]:
                page_id = word_img_name.split('-')[0]
                word_img = img_io.imread(os.path.join(path, label, word_img_name))
                words.append((1 - word_img.astype(np.float32) / 255.0, label, page_id))

        self.words = words

        # compute a mapping from class string to class id
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([elem[1] for elem in words])

        # extract unigrams from train split
        unigrams = [chr(i) for i in chain(range(ord('a'), ord('z') + 1), range(ord('0'), ord('9') + 1))]
        # unigrams = get_unigrams_from_strings(word_strings=[elem[1] for elem in words])
        # create embedding for the word_list
        self.word_embeddings = None
        word_strings = [elem[1] for elem in words]

        self.word_embeddings = build_phoc_descriptor(words=word_strings,
                                                     phoc_unigrams=unigrams,
                                                     unigram_levels=phoc_unigram_levels)

        self.word_embeddings = self.word_embeddings.astype(np.float32)

        self.classes = np.unique(labels)

        self.transforms = HomographyAugmentation()

        if dataset_type == 'test':
            # create queries
            word_strings = [elem[1] for elem in self.words]
            unique_word_strings, counts = np.unique(word_strings, return_counts=True)
            qry_word_ids = unique_word_strings[np.where(counts > 1)[0]]

            query_list = np.zeros(len(word_strings), np.int8)
            qry_ids = [i for i in range(len(word_strings)) if word_strings[i] in qry_word_ids]
            query_list[qry_ids] = 1

            self.query_list = query_list
        else:
            word_strings = [elem[1] for elem in self.words]
            self.query_list = np.zeros(len(word_strings), np.int8)

        if dataset_type == 'train':
            # weights for sampling
            # train_class_ids = [self.label_encoder.transform([self.word_list[index][1]]) for index in range(len(self.word_list))]
            # word_strings = [elem[1] for elem in self.word_list]
            unique_word_strings, counts = np.unique(word_strings, return_counts=True)
            ref_count_strings = {uword: count for uword, count in zip(unique_word_strings, counts)}
            weights = [1.0 / ref_count_strings[word] for word in word_strings]
            self.weights = np.array(weights) / sum(weights)

    def embedding_size(self):
        return len(self.word_embeddings[0])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        word_img = self.words[index][0]
        if self.transforms is not None:
            word_img = self.transforms(word_img)

        # fixed size image !!!
        word_img = self._image_resize(word_img, self.fixed_image_size)

        word_img = word_img.reshape((1,) + word_img.shape)
        word_img = torch.from_numpy(word_img)
        embedding = self.word_embeddings[index]
        embedding = torch.from_numpy(embedding)
        class_id = self.label_encoder.transform([self.words[index][1]])
        is_query = self.query_list[index]

        return word_img, embedding, class_id, is_query

    # fixed sized image
    @staticmethod
    def _image_resize(word_img, fixed_img_size):

        if fixed_img_size is not None:
            if len(fixed_img_size) == 1:
                scale = float(fixed_img_size[0]) / float(word_img.shape[0])
                new_shape = (int(scale * word_img.shape[0]), int(scale * word_img.shape[1]))

            if len(fixed_img_size) == 2:
                new_shape = (fixed_img_size[0], fixed_img_size[1])

            word_img = resize(image=word_img, output_shape=new_shape).astype(np.float32)

        return word_img


def build_phoc_descriptor(words, phoc_unigrams, unigram_levels,
                          bigram_levels=None, phoc_bigrams=None,
                          split_character=None, on_unknown_unigram='nothing',
                          phoc_type='phoc'):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).

    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels to use in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error', 'nothing'
        phoc_type (str): the type of the PHOC to be build. The default is the
            binary PHOC (standard version from Almazan 2014).
            Possible: phoc, spoc
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    logger = logging.getLogger('PHOCGenerator')
    if on_unknown_unigram not in ['error', 'warn', 'nothing']:
        raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams) * np.sum(bigram_levels)
    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k + 1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]

    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}

    # iterate through all the words
    for word_index, word in enumerate(tqdm(words)):
        if split_character is not None:
            word = word.split(split_character)

        n = len(word)  # pylint: disable=invalid-name
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:
                if on_unknown_unigram == 'warn':
                    logger.warn('The unigram \'%s\' is unknown, skipping this character', char)
                    continue
                elif on_unknown_unigram == 'error':
                    logger.fatal('The unigram \'%s\' is unknown', char)
                    raise ValueError()
                else:
                    continue
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(
                            phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        if phoc_type == 'phoc':
                            phocs[word_index, feat_vec_index] = 1
                        elif phoc_type == 'spoc':
                            phocs[word_index, feat_vec_index] += 1
                        else:
                            raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
        # add bigrams
        if phoc_bigrams is not None:
            ngram_features = np.zeros(len(phoc_bigrams) * np.sum(bigram_levels))
            ngram_occupancy = lambda k, n: [float(k) / n, float(k + 2) / n]
            for i in range(n - 1):
                ngram = word[i:i + 2]
                if phoc_bigrams.get(ngram, 0) == 0:
                    continue
                occ = ngram_occupancy(i, n)
                for level in bigram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        overlap_size = size(overlap(occ, region_occ)) / size(occ)
                        if overlap_size >= 0.5:
                            if phoc_type == 'phoc':
                                ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] = 1
                            elif phoc_type == 'spoc':
                                ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] += 1
                            else:
                                raise ValueError('The phoc_type \'%s\' is unknown' % phoc_type)
            phocs[word_index, -ngram_features.shape[0]:] = ngram_features
    return phocs
