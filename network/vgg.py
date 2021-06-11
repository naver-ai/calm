import torch.nn as nn
from torch.utils.model_zoo import load_url

from .universal import UniversalProcess
from .util import remove_layer
from .util import initialize_weights

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}

configs_dict = {
    '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
              512, 'M', 512, 512, 512],
    '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
              512, 512, 512, 512],
}


class Vgg(nn.Module):
    def __init__(self, features, use_bn=False, num_classes=1000,
                 attribution_method='CAM', **kwargs):
        super(Vgg, self).__init__()
        self.num_classes = num_classes
        self.attribution_method = attribution_method
        self.process = UniversalProcess(attribution_method)
        self.use_bn = use_bn

        self.features = features
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        if use_bn:
            self.bn6 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=False)

        self.conv_last = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.attention = self._get_attention()

        initialize_weights(self.modules(), init_mode='he')
        self.process.set_layers(self.conv_last, self.attention)

    def forward(self, x, labels=None, superclass=None, return_cam=False):
        x = self.features(x)
        x = self.conv6(x)
        if self.use_bn:
            x = self.bn6(x)
        f_former = self.relu(x)
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

    def _get_attention(self):
        return nn.Sequential(
            nn.Conv2d(512 * self.block.expansion, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=False)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model


def load_pretrained_model(model, use_bn, path=None):
    model_name = 'vgg16_bn' if use_bn else 'vgg16'
    state_dict = load_url(model_urls[model_name], progress=True)

    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=False)
    return model


def make_layers(cfg, use_bn, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if use_bn:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(pretrained=False, pretrained_path=None, use_bn=False, **kwargs):
    config_key = '28x28' if kwargs['large_feature_map'] else '14x14'
    layers = make_layers(configs_dict[config_key], use_bn, **kwargs)
    model = Vgg(layers, use_bn, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, use_bn, path=pretrained_path)
    return model
