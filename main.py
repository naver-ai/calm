"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim

from collections import OrderedDict
from config import get_configs
from data_loaders import get_data_loader
from logger import load_logger
from util import string_contains_any
import network
from network.util import remove_layer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class PerformanceMeter(object):
    def __init__(self):
        self.current_value = None
        self.value_per_epoch = []

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]


class Trainer(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = ('train', 'val', 'test')
    _RESIZE_LENGTH = 224
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
        "OpenImages": 100,
    }
    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['conv1', 'layer1', 'layer2', 'layer3'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }
    _FEATURE_PARAM_LAYER_PATTERNS_FINETUNE = {
        'vgg': ['features.', 'conv6', 'bn6'],
        'resnet': ['conv1', 'bn1', 'layer1', 'layer2',
                   'layer3', 'layer4'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4', 'SPG_A3'],
    }

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.args = get_configs()

        self.score_map_process = self.args.score_map_process
        self.score_map_method = self.args.score_map_method
        self.norm_type = self.args.norm_type
        self.threshold_type = self.args.threshold_type
        set_random_seed(self.args.seed)
        print(self.args)
        self._IOU_THRESHOLDS = self.args.iou_thresholds
        self.eval_performance_meters = self._make_eval_dict()
        self.num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        self.reporter = self.args.reporter
        self.model = self._set_model()
        if self.args.use_load_checkpoint:
            self.load_checkpoint('last', self.args.load_checkpoint)
        self.criterion = self._set_criterion()
        self.optimizer = self._set_optimizer()
        self.loc_performance_per_tau = dict()

        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
        )

        self.logger = load_logger(self.args.logger_type)

    def _make_eval_dict(self):
        _EVAL_METRICS = ['loss', 'cls'] + \
                        [f'cue_{i}' for i in [1, 2, 3]]
        eval_dict = {split: {} for split in self._SPLITS}
        for split in self._SPLITS:
            for metric in _EVAL_METRICS:
                eval_dict[split][metric] = PerformanceMeter()
        return eval_dict

    def _set_model(self):
        print("Loading model {}".format(self.args.architecture))
        model = network.__dict__[self.args.architecture](
            dataset_name=self.args.dataset_name,
            pretrained=self.args.pretrained,
            num_classes=self.num_classes,
            large_feature_map=self.args.large_feature_map,
            use_bn=self.args.use_bn,
            attribution_method=self.args.attribution_method,
            pretrained_path=self.args.pretrained_path,
        )
        model = model.to(self.device)
        print(model, '\n')
        return model

    def _set_criterion(self):
        criterion = nn.NLLLoss().to(self.device)
        return criterion

    def _set_optimizer(self):
        param_features = []
        param_classifiers = []

        def param_features_substring_list(architecture):
            patterns = self._FEATURE_PARAM_LAYER_PATTERNS_FINETUNE \
                if self.args.is_different_checkpoint \
                else self._FEATURE_PARAM_LAYER_PATTERNS
            for key in patterns:
                if architecture.startswith(key):
                    return patterns[key]
            raise KeyError("Fail to recognize the architecture {}"
                           .format(self.args.architecture))

        for name, parameter in self.model.named_parameters():
            if string_contains_any(
                    name,
                    param_features_substring_list(self.args.architecture)):
                param_features.append(parameter)
            else:
                param_classifiers.append(parameter)

        optimizer = torch.optim.SGD([
            {'params': param_features, 'lr': self.args.lr},
            {'params': param_classifiers,
             'lr': self.args.lr * self.args.lr_classifier_ratio}],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True)
        return optimizer

    def _training(self, images, target, image_id=None):
        output_dict = self.model(images)
        probs = output_dict['probs']
        features = output_dict['features']
        loss = self.criterion(features, target)
        return probs, loss

    def train(self, split):
        self.model.train()
        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        for batch_idx, (images, target, image_id) in enumerate(loader):
            images = images.to(self.device)
            target = target.to(self.device)

            if batch_idx % max(int(len(loader) / 10), 1) == 0:
                print(" iteration ({} / {})".format(batch_idx + 1, len(loader)))

            probs, loss = self._training(images, target, image_id)
            pred = probs.argmax(dim=1).detach()

            total_loss += loss.item() * images.size(0)
            num_correct += (pred == target).sum().item()
            num_images += images.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_average = total_loss / float(num_images)
        classification_acc = num_correct / float(num_images) * 100
        self.eval_performance_meters[split]['cls'].update(
            classification_acc)
        self.eval_performance_meters[split]['loss'].update(loss_average)

        return dict(cls=classification_acc,
                    loss=loss_average)

    def print_performances(self):
        for split in self._SPLITS:
            for metric in self.eval_performance_meters[split].keys():
                current_performance = \
                    self.eval_performance_meters[split][metric].current_value
                if current_performance is not None:
                    print("Split {}, metric {}: {}".format(
                        split, metric, current_performance))

    def save_performances(self):
        log_path = os.path.join(self.args.log_folder, 'performance_log.pickle')
        with open(log_path, 'wb') as f:
            pickle.dump(self.eval_performance_meters, f)

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids) in enumerate(loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            with torch.no_grad():
                output_dict = self.model(images)
            probs = output_dict['probs']
            pred = probs.argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate_cls(self, split):
        self.model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        self.eval_performance_meters[split]['cls'].update(accuracy)

    def _torch_save_model(self, filename, epoch):
        torch.save({'architecture': self.args.architecture,
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                   os.path.join(self.args.log_folder, filename))

    def save_checkpoint(self, epoch):
        self._torch_save_model(
            self._CHECKPOINT_NAME_TEMPLATE.format('last'), epoch)

    def report_train(self, train_performance, epoch, split='train'):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        reporter_instance.add(
            key='{split}/cls'.format(split=split),
            val=train_performance['cls'])
        reporter_instance.add(
            key='{split}/loss'.format(split=split),
            val=train_performance['loss'])
        reporter_instance.write()

        self.logger.report(msg_dict={'step': epoch,
                                     f'{split}/loss': train_performance['loss'],
                                     f'{split}/cls': train_performance['cls']})

    def report(self, epoch, split):
        reporter_instance = self.reporter(self.args.reporter_log_root, epoch)
        msg_dict = {'step': epoch}
        for metric in self.eval_performance_meters[split].keys():
            reporter_instance.add(
                key='{split}/{metric}'
                    .format(split=split, metric=metric),
                val=self.eval_performance_meters[split][metric].current_value)
            msg_dict[f'{split}/{metric}'] = \
                self.eval_performance_meters[split][metric].current_value
        reporter_instance.write()
        self.logger.report(msg_dict=msg_dict)

    def adjust_learning_rate(self, epoch):
        if epoch != 0 and epoch % self.args.lr_decay_frequency == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_type, load_checkpoint=None):
        checkpoint_path = self._make_ckpt_path(checkpoint_type, load_checkpoint)

        if os.path.isfile(checkpoint_path):
            if not torch.cuda.is_available():
                ckpt = torch.load(checkpoint_path,
                                  map_location=torch.device('cpu'))
                if self.args.is_different_checkpoint:
                    state_dict = remove_layer(ckpt['state_dict'], 'conv_last')
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    self.model.load_state_dict(ckpt['state_dict'], strict=True)
            else:
                checkpoint = torch.load(checkpoint_path)
                state_dict = self._remove_module(checkpoint)
                if self.args.is_different_checkpoint:
                    state_dict = remove_layer(state_dict, 'conv_last')
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    self.model.load_state_dict(state_dict, strict=True)
            print("Check {} loaded.\n".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))

    def _remove_module(self, checkpoint):
        state_dict = OrderedDict()
        for key, value in checkpoint['state_dict'].items():
            state_dict[key[7:] if key.startswith('module') else key] = value
        return state_dict

    def _make_ckpt_path(self, checkpoint_type, load_checkpoint):
        if load_checkpoint is None:
            checkpoint_path = os.path.join(
                self.args.log_folder,
                self._CHECKPOINT_NAME_TEMPLATE.format(checkpoint_type))
        else:
            checkpoint_path = os.path.join(load_checkpoint,
                                           'last_checkpoint.pth.tar')
        return checkpoint_path


def main():
    trainer = Trainer()
    print(trainer.device)

    print("===========================================================")
    if trainer.args.is_train:
        for epoch in range(trainer.args.epochs):
            print("Start epoch {} ...".format(epoch + 1))
            trainer.adjust_learning_rate(epoch + 1)
            train_performance = trainer.train(split='train')
            trainer.report_train(train_performance, epoch + 1, split='train')
            trainer.evaluate_cls(split='val')
            trainer.report(epoch + 1, split='val')
            trainer.print_performances()
            print("Epoch {} done.".format(epoch + 1))

        trainer.save_checkpoint(trainer.args.epochs)

    print("===========================================================")
    print("Final epoch evaluation on test set ...")
    trainer.evaluate_cls(split='test')
    trainer.report(trainer.args.epochs, split='test')
    trainer.print_performances()

    trainer.save_performances()
    trainer.logger.finalize_log()


if __name__ == '__main__':
    main()
