import sys
sys.path.append('../../')
import torch_ard as nn_ard
from torch import nn
import torch.nn.functional as F
import torch

class DenseModelARD(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=150, activation=None):
        super(DenseModelARD, self).__init__()
        self.l1 = nn_ard.LinearARD(input_shape, hidden_size)
        self.l2 = nn_ard.LinearARD(hidden_size, output_shape)
        self.activation = activation
        self._init_weights()

    def forward(self, input):
        x = input.to(self.device)
        x = self.l1(x)
        x = nn.functional.tanh(x)
        x = self.l2(x)
        if self.activation: x = self.activation(x)
        return x

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    @property
    def device(self):
        return next(self.parameters()).device


class DenseModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=150, activation=None):
        super(DenseModel, self).__init__()
        self.l1 = nn.Linear(input_shape, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_shape)
        self.activation = activation
        self._init_weights()

    def forward(self, input):
        x = input.to(self.device)
        x = self.l1(x)
        x = nn.functional.tanh(x)
        x = self.l2(x)
        if self.activation: x = self.activation(x)
        return x

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    @property
    def device(self):
        return next(self.parameters()).device

class LeNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.l1 = nn.Linear(50*5*5, 500)
        self.l2 = nn.Linear(500, output_shape)
        self._init_weights()

    def forward(self, x):
        out = F.relu(self.conv1(x.to(self.device)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.l1(out))
        return self.l2(out)
        # return F.log_softmax(self.l2(out), dim=1)

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    @property
    def device(self):
        return next(self.parameters()).device

class LeNetARD(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LeNetARD, self).__init__()
        self.conv1 = nn_ard.Conv2dARD(input_shape, 20, 5)
        self.conv2 = nn_ard.Conv2dARD(20, 50, 5)
        self.l1 = nn_ard.LinearARD(50*5*5, 500)
        self.l2 = nn_ard.LinearARD(500, output_shape)
        self._init_weights()

    def forward(self, input):
        out = F.relu(self.conv1(input.to(self.device)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.l1(out))
        return self.l2(out)
        # return F.log_softmax(self.l2(out), dim=1)

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

    @property
    def device(self):
        return next(self.parameters()).device

class VGG(nn.Module):
    def __init__(self, builder, features):
        super(VGG, self).__init__()
        self.features = features
        num_classes = 10
        self.linear = nn_ard.LinearARD(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x.squeeze()


def make_layers(cfg, builder, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = builder.conv3x3(in_channels, v) #nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = nn_ard.Conv2dARD(in_channels, v, 3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, eps=1e-5), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, batch_norm, builder):
    model = VGG(builder, make_layers(cfgs[cfg], builder, batch_norm=batch_norm))
    return model


def vgg11_new_fc():
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('A', False, get_builder())


def vgg11_bn_new_fc():
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('A', True, get_builder())


def vgg13_fc():
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('B', False, get_builder())


def vgg13_bn_fc():
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('B', True, get_builder())


def vgg16_new_fc():
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('D', False, get_builder())


def vgg16_bn_fc():
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('D', False, get_builder())


def vgg19_new_fc():
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('E', False, get_builder())


def vgg19_bn_new_fc():
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('E', True, get_builder())



class LeNet_MNIST(LeNet):
    def __init__(self, input_shape, output_shape):
        super(LeNet_MNIST, self).__init__(input_shape, output_shape)
        self.l1 = nn.Linear(50*4*4, 500)
        super(LeNet_MNIST, self)._init_weights()

class LeNetARD_MNIST(LeNetARD):
    def __init__(self, input_shape, output_shape):
        super(LeNetARD_MNIST, self).__init__(input_shape, output_shape)
        self.l1 = nn_ard.LinearARD(50*4*4, 500)
        super(LeNetARD_MNIST, self)._init_weights()
