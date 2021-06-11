"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import argparse
import munch
import importlib
import os

from os.path import join as ospj
import shutil

from util import Logger

_DATASET_NAMES = ('CUB', 'ILSVRC', 'OpenImages')
_ARCHITECTURE_NAMES = ('vgg16', 'resnet50', 'inception_v3')
_ATTRIBUTION_METHODS = ('CAM', 'CALM-EM', 'CALM-ML')
_SCORE_MAP_METHOD_NAMES = ('activation_map', 'backprop')
_SCORE_MAP_PROCESS_NAMES = (
    'vanilla', 'vanilla-saliency', 'vanilla-superclass',
    'jointll', 'jointll-superclass', 'jointll-superclass-mean',
    'gtcond', 'gtcond-superclass', 'gtcond-superclass-mean',
    'saliency', 
    'input_grad', 'integrated_grad', 'smooth_grad', 'var_grad')
_NORM_TYPES = ('max', 'minmax', 'clipping')
_THRESHOLD_TYPES = ('even', 'log')
_SPLITS = ('train', 'val', 'test')
_LOGGER_TYPE = ('PythonLogger')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def configure_data_paths(args):
    train, val, test = set_data_path(
        dataset_name=args.dataset_name,
        data_root=args.data_root
    )
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def set_data_path(dataset_name, data_root):
    if dataset_name == 'ILSVRC':
        train = test = ospj(data_root, dataset_name)
        val = ospj(data_root, 'ImageNetV2')
    elif dataset_name == 'CUB':
        train = test = ospj(data_root, dataset_name, 'images')
        val = ospj(data_root, 'CUBV2')
    elif dataset_name == 'OpenImages':
        train = val = test = ospj(data_root, dataset_name)
    else:
        raise ValueError("Dataset {} unknown.".format(dataset_name))
    return train, val, test


def configure_mask_root(args):
    mask_root = ospj(args.mask_root, 'OpenImages')
    return mask_root


def configure_log_folder(args):
    log_folder = ospj(args.save_root, args.experiment_name)

    if os.path.isdir(log_folder):
        if args.override_cache:
            shutil.rmtree(log_folder, ignore_errors=True)
        else:
            raise RuntimeError("Experiment with the same name exists: {}"
                               .format(log_folder))
    os.makedirs(log_folder)
    return log_folder


def configure_log(args):
    log_file_name = ospj(args.log_folder, 'log.log')
    Logger(log_file_name)


def configure_reporter(args):
    reporter = importlib.import_module('util').Reporter
    reporter_log_root = ospj(args.log_folder, 'reports')
    if not os.path.isdir(reporter_log_root):
        os.makedirs(reporter_log_root)
    return reporter, reporter_log_root


def configure_pretrained_path(args):
    pretrained_path = None
    return pretrained_path


def get_configs():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--seed', type=int)
    parser.add_argument('--experiment_name', type=str, default='result')
    parser.add_argument('--override_cache', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--use_load_checkpoint', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='folder name for loading ckeckpoint')
    parser.add_argument('--is_different_checkpoint', type=str2bool, 
                        nargs='?', const=True, default=False)
    parser.add_argument('--save_root', type=str, default='save')
    parser.add_argument('--logger_type', type=str,
                        default='PythonLogger', choices=_LOGGER_TYPE)

    # Data
    parser.add_argument('--dataset_name', type=str, default='CUB',
                        choices=_DATASET_NAMES)
    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='dataset/',
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default='metadata/')
    parser.add_argument('--mask_root', metavar='/PATH/TO/MASKS',
                        default='dataset/',
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')

    # Setting
    parser.add_argument('--architecture', default='resnet18',
                        choices=_ARCHITECTURE_NAMES,
                        help='model architecture: ' +
                             ' | '.join(_ARCHITECTURE_NAMES) +
                             ' (default: resnet18)')
    parser.add_argument('--attribution_method', type=str, default='CAM',
                        choices=_ATTRIBUTION_METHODS)
    parser.add_argument('--is_train', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Use pre_trained model.')
    parser.add_argument('--cam_curve_interval', type=float, default=.001,
                        help='CAM curve interval')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='input crop size')

    # Common hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini-batch size (default: 256), this is the total'
                             'batch size of all GPUs on the current node when'
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate', dest='lr')
    parser.add_argument('--lr_decay_frequency', type=int, default=30,
                        help='How frequently do we decay the learning rate?')
    parser.add_argument('--lr_classifier_ratio', type=float, default=10,
                        help='Multiplicative factor on the classifier layer.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--use_bn', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--iou_thresholds', nargs='+',
                        type=int, default=[30, 50, 70])

    # Method-specific hyperparameters
    parser.add_argument('--smoothing_ksize', type=int, default=1)
    parser.add_argument('--score_map_method', type=str, default='activation_map',
                        choices=_SCORE_MAP_METHOD_NAMES)
    parser.add_argument('--score_map_process', type=str, default='vanilla',
                        choices=_SCORE_MAP_PROCESS_NAMES)
    parser.add_argument('--norm_type', default='minmax', type=str,
                        choices=_NORM_TYPES)
    parser.add_argument('--threshold_type', default='even', type=str,
                        choices=_THRESHOLD_TYPES)
    parser.add_argument('--smooth_grad_nr_iter', type=int, default=50,
                        help='SmoothGrad number of sampling')
    parser.add_argument('--smooth_grad_sigma', type=float, default=4.0,
                        help='SmoothGrad sigma multiplier')
    parser.add_argument('--integrated_grad_nr_iter', type=int, default=50,
                        help='IntegratedGradient number of steps')
    args = parser.parse_args()

    args.log_folder = configure_log_folder(args)
    configure_log(args)

    args.data_root = args.data_root.strip('"')
    args.data_paths = configure_data_paths(args)
    args.metadata_root = ospj(args.metadata_root, args.dataset_name)
    args.mask_root = configure_mask_root(args)
    args.reporter, args.reporter_log_root = configure_reporter(args)
    args.pretrained_path = configure_pretrained_path(args)

    return args
