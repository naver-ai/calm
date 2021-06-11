"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import json
import math
import logging
import numbers
import numpy as np
import os
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import score_map_method

_NUM_CLASSES_MAPPING = {
    "CUB": 200,
    "ILSVRC": 1000,
    "OpenImages": 100,
}
_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


def smoothing(score_maps, image_size, smoothing_ksize):
    score_maps = score_maps.unsqueeze(1)
    if score_maps.shape[2:] == image_size:
        sigma = 0.3 * ((smoothing_ksize - 1) * 0.5 - 1) + 0.8
        smoothing = GaussianSmoothing(1, smoothing_ksize, sigma)
        score_maps = F.pad(score_maps,
                           compute_padding(
                               (smoothing_ksize, smoothing_ksize)),
                           mode='reflect')
        score_maps = smoothing(score_maps)
    return score_maps.squeeze(1)


def compute_padding(smoothing_ksize):
    computed = [k // 2 for k in smoothing_ksize]

    return [computed[1] - 1 if smoothing_ksize[0] % 2 == 0 else computed[1],
            computed[1],
            computed[0] - 1 if smoothing_ksize[1] % 2 == 0 else computed[0],
            computed[0]]


def get_score_map(process, method):
    return score_map_method.__dict__[
        process if '_grad' in process else method]


def get_scoremaps(tester, inputs, targets, dataset_name):
    smoothing_ksize = 1
    score_map_process = tester.args.score_map_process
    score_map_method_name = tester.args.score_map_method

    scoremaps = get_score_map(
        score_map_process, score_map_method_name)(
        model=tester.model,
        images=inputs,
        targets=targets,
        num_classes=_NUM_CLASSES_MAPPING[dataset_name],
        score_map_method=score_map_method_name,
        score_map_process=score_map_process,
        method=score_map_method.__dict__[score_map_method_name],
        smooth_grad_nr_iter=tester.args.smooth_grad_nr_iter,
        smooth_grad_sigma=tester.args.smooth_grad_sigma,
        integrated_grad_nr_iter=tester.args.integrated_grad_nr_iter,
    )
    if score_map_method_name == 'backprop':
        scoremaps = smoothing(scoremaps, inputs.shape[2:], smoothing_ksize)
    return scoremaps.cpu().detach().numpy()


def set_logger(name=None, level=None, fmt=None, datefmt=None, **kwargs):
    logger = logging.getLogger(name)
    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    # create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # create formatter and add it to the handlers
    if fmt is None:
        fmt = '[%(asctime)s] %(message)s'
    if datefmt is None:
        datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def string_contains_any(string, substring_list):
    for substring in substring_list:
        if substring in string:
            return True
    return False


class Reporter(object):
    def __init__(self, reporter_log_root, epoch):
        self.log_file = os.path.join(reporter_log_root, str(epoch))
        self.epoch = epoch
        self.report_dict = {
            'summary': True,
            'step': self.epoch,
        }

    def add(self, key, val):
        self.report_dict.update({key: val})

    def write(self):
        log_file = self.log_file
        while os.path.isfile(log_file):
            log_file += '_'
        with open(log_file, 'w') as f:
            f.write(json.dumps(self.report_dict))


def get_baseline(baseline_, inputs, device):
    B, C, H, W = inputs.size()
    if baseline_ == 'mean':
        baseline = inputs.mean(dim=[2, 3], keepdim=True).expand(B, C, H,
                                                                W).contiguous()
    elif baseline_ == 'noise':
        baseline = inputs.mean(dim=[2, 3], keepdim=True).expand(B, C, H,
                                                                W).contiguous()
        baseline += torch.randn(baseline.size()).to(device) * 0.2
    elif baseline_ == 'blur':
        kernel_size, kernel_std = 31, 15
        smoothing = GaussianSmoothing(C, kernel_size, kernel_std).to(device)
        baseline = smoothing(
            F.pad(inputs, [kernel_size // 2] * 4, mode='reflect'))
    elif baseline_ == 'zero':
        baseline = torch.zeros_like(inputs).to(device)
    return baseline


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors.
                                  Output will have this number of channels as
                                  well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        self.conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[dim]

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.cuda(), groups=self.groups)


def get_topk_to_zero_mask(mu, k, device):
    B, _, H, W = mu.size()
    mu_flat = mu.view(B, -1)
    mu_sort_idxs = mu_flat.topk(k=H * W, dim=1)[1]
    mu_topk_idxs = mu_sort_idxs[:, :int(k * H * W)]

    # pixels of topk values to 0
    mask = torch.ones(B, H * W).to(device)
    mask.scatter_(1, mu_topk_idxs, 0.)
    mask = mask.view(B, 1, H, W)
    return mask


def random_mask_generator(inputs, device, k):
    B, _, H, W = inputs.size()
    randk = [np.random.choice(H * W, size=int(H * W * k), replace=False) for _
             in range(B)]
    randk = torch.tensor(np.array(randk)).to(device)
    masks = torch.ones_like(inputs[:, 0, :, :])
    masks.view(B, -1).scatter_(1, randk, 0)
    masks = masks.view(B, 1, H, W)
    return masks


def resize_scoremaps(scoremaps, dtype='tensor'):
    """
    Inputs:
        scoremaps: np.array, shape = (B, 28, 28)
        dtype: 'tensor' or 'numpy'
    Returns:
        scoremaps: np.array or torch.tensor, size = (B, 224, 224)
    """
    scoremaps_list = []
    for scoremap in scoremaps:
        scoremap = resize_scoremap(scoremap)
        scoremap = torch.tensor(scoremap) if dtype == 'tensor' else scoremap
        scoremaps_list.append(scoremap)
    scoremaps = stack_scoremap(scoremaps_list, dtype)
    return scoremaps


def stack_scoremap(scoremaps_list, dtype):
    if dtype == 'tensor':
        scoremaps = torch.stack(scoremaps_list, dim=0).unsqueeze(1)
    elif dtype == 'numpy':
        scoremaps = np.stack(scoremaps_list, axis=0)
    else:
        raise ValueError(f'{dtype} is not available.')
    return scoremaps


def resize_scoremap(score_map,
                    image_size=(224, 224),
                    transform=cv2.INTER_LINEAR):
    if score_map.shape != image_size:
        score_map = cv2.resize(score_map, image_size,
                               interpolation=transform)
    return score_map



def normalize_scoremap(score_map, norm_type):
    """
    Args:
        score_map: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(score_map).any():
        return np.zeros_like(score_map)
    if score_map.min() == score_map.max():
        return np.zeros_like(score_map)
    if norm_type == 'max':
        score_map /= score_map.max()
    elif norm_type == 'minmax':
        score_map -= score_map.min()
        score_map /= score_map.max()
    elif norm_type == 'clipping':
        pass
    score_map[score_map > 1.] = 1.
    score_map[score_map < 0.] = 0.
    return score_map


class SaveScoreMap(object):
    def __init__(self):
        self.cnt = 0

    def save(self, image, saliency, method_name):
        path = os.path.join(method_name, 'saliency_{}.jpg'.format(self.cnt))
        if len(image.size()) == 4:
            image = image.squeeze(0)
        image = t2n(image).transpose(1, 2, 0)
        image = ((image * _IMAGENET_STDDEV) + _IMAGENET_MEAN)
        saliency = np.expand_dims(saliency, axis=0)
        saliency = np.repeat(saliency, 3, axis=0)
        saliency = saliency.transpose(1, 2, 0)
        output = np.concatenate([image, saliency], axis=1)
        output = (output * 255).astype(np.uint8)
        output = Image.fromarray(output)
        output.save(path)
        self.cnt += 1


def get_threshold_list(cam_curve_interval, threshold_type):
    if threshold_type == 'log':
        threshold_list = np.arange(0, 7, cam_curve_interval * 7)
        threshold_list = [10 ** (th * (-1)) for th in threshold_list[::-1]]
    elif threshold_type == 'even':
        threshold_list = list(np.arange(0, 1, cam_curve_interval))
    else:
        raise ValueError(f'Invalid threshold_type argument: {threshold_type}')
    return threshold_list
