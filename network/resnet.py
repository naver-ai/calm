import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

from .universal import UniversalProcess
from .util import remove_layer
from .util import initialize_weights

__all__ = ['resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, attribution_method='CAM',
                 **kwargs):
        super(ResNet, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64
        self.num_classes = num_classes
        self.block = block
        self.attribution_method = attribution_method
        self.process = UniversalProcess(attribution_method)

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.conv_last = nn.Conv2d(
            512 * block.expansion, num_classes, kernel_size=1)
        self.attention = self._get_attention()

        initialize_weights(self.modules(), init_mode='xavier')
        self.process.set_layers(self.conv_last, self.attention)

    def forward(self, x, labels=None, superclass=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f_former = self.layer4(x)
        f = self.conv_last(f_former)

        if return_cam:
            f = f.detach()
            f_former = f_former.detach()
            cams = self.process.get_cam(f, f_former, labels,
                                        superclass, return_cam)
            return cams
        else:
            inputs = self.process.input_for_loss(f, f_former)
            return {'logits': inputs['logits'],
                    'probs': inputs['probs'],
                    'features': inputs['inputs']}

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(
            self.inplanes, block, planes, stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def _get_attention(self):
        return nn.Sequential(
                nn.Conv2d(512 * self.block.expansion, 1, kernel_size=1),
                nn.ReLU(inplace=True),
        )


def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


def load_pretrained_model(model, path=None):
    strict_rule = True

    if path:
        state_dict = torch.load(os.path.join(path, 'resnet50.pth'))
    else:
        state_dict = load_url(model_urls['resnet50'], progress=True)

    state_dict = remove_layer(state_dict, 'fc')
    strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def resnet50(pretrained=False, pretrained_path=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model = load_pretrained_model(model, path=pretrained_path)
    return model
