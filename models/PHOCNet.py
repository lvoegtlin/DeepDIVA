import torch.nn as nn
import torch.nn.functional as F
import torch


class PHOCNet(nn.Module):
    '''
    Network class for generating PHOCNet and TPP-PHOCNet architectures

    Source: https://github.com/georgeretsi/pytorch-phocnet/tree/master/experiments/cnn_ws_experiments
    '''

    def __init__(self, output_channels=1000, input_channels=1, gpp_type='spp', pooling_levels=3, pool_type='max_pool',
                 **kwargs):
        super(PHOCNet, self).__init__()

        # some sanity checks
        if gpp_type not in ['spp', 'tpp', 'gpp']:
            raise ValueError('Unknown pooling_type. Must be either \'gpp\', \'spp\' or \'tpp\'')
        # set up Conv Layers
        self.conv1_1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # create the spatial pooling layer
        self.pooling_layer_fn = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type=pool_type)
        pooling_output_size = self.pooling_layer_fn.pooling_output_size
        self.fc5 = nn.Linear(pooling_output_size, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, output_channels)

        self.init_weights()

    def forward(self, x):
        y = F.relu(self.conv1_1(x))
        y = F.relu(self.conv1_2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2, padding=0)
        y = F.relu(self.conv2_1(y))
        y = F.relu(self.conv2_2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2, padding=0)
        y = F.relu(self.conv3_1(y))
        y = F.relu(self.conv3_2(y))
        y = F.relu(self.conv3_3(y))
        y = F.relu(self.conv3_4(y))
        y = F.relu(self.conv3_5(y))
        y = F.relu(self.conv3_6(y))
        y = F.relu(self.conv4_1(y))
        y = F.relu(self.conv4_2(y))
        y = F.relu(self.conv4_3(y))

        y = self.pooling_layer_fn.forward(y)
        y = F.relu(self.fc5(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = F.relu(self.fc6(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.fc7(y)
        return y

    def init_weights(self):
        self.apply(PHOCNet._init_weights_he)


    '''
    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            #nn.init.kaiming_normal(m.weight.data)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n)**(1/2.0))
            if hasattr(m, 'bias'):
                nn.init.constant(m.bias.data, 0)
    '''

    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
        if isinstance(m, nn.Linear):
            n = m.out_features
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            nn.init.constant(m.bias.data, 0)


class GPP(nn.Module):
    """
    Source: https://github.com/georgeretsi/pytorch-phocnet/tree/master/experiments/cnn_ws_experiments
    """

    def __init__(self, gpp_type='tpp', levels=3, pool_type='max_pool'):
        super(GPP, self).__init__()

        if gpp_type not in ['spp', 'tpp', 'gpp']:
            raise ValueError('Unknown gpp_type. Must be either \'spp\', \'tpp\', \'gpp\'')

        if pool_type not in ['max_pool', 'avg_pool']:
            raise ValueError('Unknown pool_type. Must be either \'max_pool\', \'avg_pool\'')

        if gpp_type == 'spp':
            self.pooling_output_size = sum([4 ** level for level in range(levels)]) * 512
        elif gpp_type == 'tpp':
            self.pooling_output_size = (2 ** levels - 1) * 512
        if gpp_type == 'gpp':
            self.pooling_output_size = sum([h * w for h in levels[0] for w in levels[1]]) * 512

        self.gpp_type = gpp_type
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, input_x):

        if self.gpp_type == 'spp':
            return self._spatial_pyramid_pooling(input_x, self.levels)
        if self.gpp_type == 'tpp':
            return self._temporal_pyramid_pooling(input_x, self.levels)
        if self.gpp_type == 'gpp':
            return self._generic_pyramid_pooling(input_x, self.levels)

    def _pyramid_pooling(self, input_x, output_sizes):
        pyramid_level_tensors = []
        for tsize in output_sizes:
            if self.pool_type == 'max_pool':
                pyramid_level_tensor = F.adaptive_max_pool2d(input_x, tsize)
            if self.pool_type == 'avg_pool':
                pyramid_level_tensor = F.adaptive_avg_pool2d(input_x, tsize)
            pyramid_level_tensor = pyramid_level_tensor.view(input_x.size(0), -1)
            pyramid_level_tensors.append(pyramid_level_tensor)
        return torch.cat(pyramid_level_tensors, dim=1)

    def _spatial_pyramid_pooling(self, input_x, levels):

        output_sizes = [(int(2 ** level), int(2 ** level)) for level in range(levels)]

        return self._pyramid_pooling(input_x, output_sizes)

    def _temporal_pyramid_pooling(self, input_x, levels):

        output_sizes = [(1, int(2 ** level)) for level in range(levels)]

        return self._pyramid_pooling(input_x, output_sizes)

    def _generic_pyramid_pooling(self, input_x, levels):

        levels_h = levels[0]
        levels_w = levels[1]
        output_sizes = [(int(h), int(w)) for h in levels_h for w in levels_w]

        return self._pyramid_pooling(input_x, output_sizes)
