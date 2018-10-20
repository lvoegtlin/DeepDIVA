# Utils
from __future__ import print_function

import logging
import os

# Torch
import torch
import torchvision.transforms as transforms

# DeepDIVA
from datasets.phocnet_gw_dataset import load_dataset
from template.setup import _dataloaders_from_datasets, _load_mean_std_from_file


def setup_dataloaders(dataset_folder, batch_size, workers, inmem, phoc_unigram_levels, **kwargs):
    """
    Set up the dataloaders for the specified datasets.

    Parameters
    ----------
    model_expected_input_size : tuple
        Specify the height and width that the model expects.
    dataset_folder : string
        Path string that points to the three folder train/val/test. Example: ~/../../data/svhn

    batch_size : int
        Number of datapoints to process at once
    workers : int
        Number of workers to use for the dataloaders
    inmem : boolean
        Flag : if False, the dataset is loaded in an online fashion i.e. only file names are stored
        and images are loaded on demand. This is slower than storing everything in memory.


    Returns
    -------
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
        Dataloaders for train, val and test.
    """

    # Recover dataset name
    dataset = os.path.basename(os.path.normpath(dataset_folder))
    logging.info('Loading {} from:{}'.format(dataset, dataset_folder))

    ###############################################################################################
    # Load the dataset splits as images
    train_ds, test_ds = load_dataset(dataset_folder=dataset_folder,
                                     phoc_unigram_levels=phoc_unigram_levels)



    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')

    standard_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds.transform = standard_transform
    test_ds.transform = standard_transform

    train_loader, test_loader = _dataloaders_from_datasets(batch_size=batch_size,
                                                                       train_ds=train_ds,
                                                                       test_ds=test_ds,
                                                                       workers=workers)
    return train_loader, test_loader, train_ds[0][1].shape[0]


def _dataloaders_from_datasets(batch_size, train_ds, test_ds, workers):
    """
    This function creates (and returns) dataloader from datasets objects

    Parameters
    ----------
    batch_size : int
        The size of the mini batch
    train_ds : data.Dataset
    test_ds : data.Dataset
        Train, validation and test splits
    workers:
        Number of workers to use to load the data.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
    val_loader : torch.utils.data.DataLoader
    test_loader : torch.utils.data.DataLoader
        The dataloaders for each split passed
    """
    # Setup dataloaders
    logging.debug('Setting up dataloaders')
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               shuffle=True,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=batch_size,
                                              num_workers=workers,
                                              pin_memory=True)
    return train_loader, test_loader
