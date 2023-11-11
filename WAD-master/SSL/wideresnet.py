import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import random
import numpy as np
import os

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']= str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        self.update_batch_stats = True

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBasicBlock(MetaModule):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(MetaBasicBlock, self).__init__()

        self.bn1 = MetaBatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv1 = MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv2 = MetaConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class MetaNetworkBlock(MetaModule):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(MetaNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(MetaModule):
    def __init__(self, depth=28, widen_factor=2, n_classes=10, dropRate=0.0, transform_fn=None, seed=0):
        # control the initialization
        seed_torch(seed)
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MetaBasicBlock
        self.conv1 = MetaConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = MetaNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = MetaNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = MetaNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = MetaBatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(0.1)
        self.fc = MetaLinear(nChannels[3], n_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MetaLinear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.transform_fn = transform_fn
    def forward(self, x):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, MetaBatchNorm2d):
                m.update_batch_stats = flag


class WNet(MetaModule):
    def __init__(self, input, hidden, output, seed):
        seed_torch(seed)
        super(WNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


class LeNet(MetaModule):
    def __init__(self, n_out):
        super(LeNet, self).__init__()

        layers = []
        layers.append(MetaConv2d(1, 6, 3, padding=1))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        layers.append(nn.ReLU())

        layers.append(MetaConv2d(6, 16, 3, padding=1))
        layers.append(nn.MaxPool2d(3, stride=2, padding=1))
        layers.append(nn.ReLU())

        layers.append(MetaConv2d(16, 120, 3, padding=1))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(MetaLinear(120 * 7 * 7, 84))
        layers.append(nn.ReLU())
        layers.append(MetaLinear(84, n_out))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120 * 7 * 7)
        x = self.fc_layers(x)
        return x

class CNN(MetaModule):
    def __init__(self, n_out):
        super(CNN, self).__init__()

        self.conv = torch.nn.Sequential(MetaConv2d(1, 16, 3, padding=1),
                                        nn.MaxPool2d(3, stride=2, padding=1),
                                        nn.ReLU(),
                                        MetaConv2d(16, 32, 3, padding=1),
                                        nn.MaxPool2d(3, stride=2, padding=1),
                                        nn.ReLU()
                                        )
        self.dense = torch.nn.Sequential(nn.Dropout(p=0.5),
                                        MetaLinear(32 * 7 * 7, n_out))
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.dense(x)
        return x