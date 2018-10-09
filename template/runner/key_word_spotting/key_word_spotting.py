import logging
import sys

# Utils
import numpy as np

# DeepDIVA
import models
# Delegated
from template.runner.key_word_spotting import evaluate, train
from template.setup import set_up_model, set_up_dataloaders
from util.misc import checkpoint, adjust_learning_rate
from template.runner.key_word_spotting.xml_io import XMLReader


class KeyWordSpotting:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr, validation_interval,
                   phocnet_train_xml_path, phocnet_test_xml_path, phocnet_use_lower_case_only, **kwargs):
        """
        TODO
        """

        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        # Setting up the dataloaders
        train_loader, val_loader, test_loader, num_classes = set_up_dataloaders(model_expected_input_size, **kwargs)

        print(kwargs.get('data_folder'))

        xml_reader = XMLReader(phocnet_use_lower_case_only)
        dataset_name, train_list, test_list = xml_reader.load_train_test_xml(
            train_xml_path=phocnet_train_xml_path,
            test_xml_path=phocnet_test_xml_path,
            img_dir=train_loader)

