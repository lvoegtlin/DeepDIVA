"""
DeepdDiva implementation of the Key Word Spotting network PHOCNet (https://github.com/ssudholt/phocnet).
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class _PHOCNet(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    """

    def __init__(self, **kwargs):
        """
        Creates the PHOCNet model from the scratch.

        Parameters
        ----------

        """
        super(_PHOCNet, self).__init__()

        self.expected_input_size = (None, None)
        self.output_num = [1, 2, 4]

        # First conv layer, First layer
        self.conv1_start = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # First conv layer, Second layer with max pooling
        self.conv1_end = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Second conv layer, first layer
        self.conv2_start = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Second conv layer, second layer with maxpooling
        self.conv2_end = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third conv layer, first layer
        self.conv3_start = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Third conv layer, second layer
        self.conv3_mid = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Third conv layer, sixth layer
        self.conv3_end = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fourth conv layer, first layer
        self.conv4_start = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fourth conv layer, second layer
        self.conv4_mid = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fourth conv layer, third layer with SSP
        self.conv4_end = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc_start = nn.Sequential(
            nn.Linear(in_features=983040, out_features=4096),
            nn.ReLU()
        )

        self.fc_mid = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU()
        )

        self.fc_end = nn.Sequential(
            nn.Linear(in_features=4096, out_features=604),
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """

        x = self.conv1_start(x)
        x = self.conv1_end(x)
        x = self.conv2_start(x)
        x = self.conv2_end(x)

        x = self.conv3_start(x)
        x = self.conv3_mid(x)
        x = self.conv3_mid(x)
        x = self.conv3_mid(x)
        x = self.conv3_mid(x)
        x = self.conv3_end(x)

        x = self.conv4_start(x)
        x = self.conv4_mid(x)
        x = self.conv4_end(x)
        x = spatial_pyramid_pool(x, 1, [int(x.size(2)), int(x.size(3))], self.output_num) # 983040
        # now we have a 1-dim vector
        x = self.fc_start(x)
        x = F.dropout2d(input=x, training=self.training, p=0.5)
        x = self.fc_mid(x)
        x = F.dropout2d(input=x, training=self.training, p=0.5)
        x = self.fc_end(x)
        x = F.sigmoid(x)

        return x


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = math.ceil((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
        w_pad = math.ceil((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:", spp.size())
        else:
            # print("size:", spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


def phocnet(**kwargs):
    """
    Returns an PHOCNet model.

    Parameters
    ----------
    """
    return _PHOCNet(**kwargs)

