"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import cv2
import os
import numpy as np
import torch
from os.path import join as ospj

from data_loaders import configure_metadata
from data_loaders import get_data_loader
from data_loaders import get_mask_paths
from util import check_scoremap_validity
from util import resize_scoremaps
from util import get_scoremaps
from util_cub_trait import CUBTrait
from main import Trainer
from util import normalize_scoremap
from util import get_threshold_list

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

NUM_DIFF_LIST = [1, 2, 3]
_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self, metadata, dataset_name, split,
                 iou_thresholds, threshold_list,
                 mask_root):
        self.metadata = metadata
        self.threshold_list = threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.mask_root = mask_root
        self.iou_thresholds = [50] if iou_thresholds is None else iou_thresholds

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


def load_mask_image(file_path, resize_size):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    return mask


def get_mask(mask_root, mask_paths, ignore_path, use_ignore=True):
    """
    Ignore mask is set as the ignore box region \setminus the ground truth
    foreground region.

    Args:
        mask_root: string.
        mask_paths: iterable of strings.
        ignore_path: string.

    Returns:
        mask: numpy.ndarray(size=(224, 224), dtype=np.uint8)
    """
    mask_all_instances = []
    for mask_path in mask_paths:
        mask_file = os.path.join(mask_root, mask_path)
        mask = load_mask_image(mask_file, (_RESIZE_LENGTH, _RESIZE_LENGTH))
        mask_all_instances.append(mask > 0.5)
    mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)

    ignore_file = os.path.join(mask_root, ignore_path)
    ignore_box_mask = load_mask_image(ignore_file,
                                      (_RESIZE_LENGTH, _RESIZE_LENGTH))
    ignore_box_mask = ignore_box_mask > 0.5

    ignore_mask = np.logical_and(ignore_box_mask,
                                 np.logical_not(mask_all_instances))

    if np.logical_and(ignore_mask, mask_all_instances).any():
        raise RuntimeError("Ignore and foreground masks intersect.")

    if use_ignore:
        return (mask_all_instances.astype(np.uint8) +
                255 * ignore_mask.astype(np.uint8))
    else:
        return mask_all_instances.astype(np.uint8)


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)

        if self.dataset_name not in ["OpenImages", "CUB"]:
            raise ValueError(
                "Mask evaluation must be performed on OpenImages or CUB.")

        if self.dataset_name == "OpenImages":
            self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.threshold_list,
                                                   [1.0, 2.0, 3.0])
        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=np.float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=np.float)

    def accumulate(self, scoremap, image_id, gt_mask=None):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        check_scoremap_validity(scoremap)
        if gt_mask is None:
            gt_mask = get_mask(self.mask_root,
                               self.mask_paths[image_id],
                               self.ignore_paths[image_id])

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(np.float)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(np.float)

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
                Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        pr_curve = [recall, precision]

        print("Mask AUC on split {}: {}".format(self.split, auc))
        return auc, pr_curve


def _get_score_map_diff(tester, inputs, class_pair, dataset_name='CUB'):
    """ actiavtion map and its difference """
    id1, id2 = class_pair
    activation_maps1 = get_scoremaps(tester, inputs.to(tester.device),
                                     torch.tensor([id1] * inputs.size(0)),
                                     dataset_name)
    activation_maps2 = get_scoremaps(tester, inputs.to(tester.device),
                                     torch.tensor([id2] * inputs.size(0)),
                                     dataset_name)
    maps_diff = activation_maps1 - activation_maps2
    maps_diff = resize_scoremaps(maps_diff, 'numpy')
    return maps_diff


def _cub_eval_inner_loop(tester, data_loader, class_pair, trait,
                         class_pair_list, evaluator):
    for inputs, targets, image_ids in data_loader:
        maps_diff = _get_score_map_diff(tester, inputs, class_pair)
        for map_diff, image_id in zip(maps_diff, image_ids):
            scoremap_abs = normalize_scoremap(
                abs(map_diff), tester.args.norm_type).astype(np.float)
            gt_mask = trait.get_pseudo_segment_mask(image_id, class_pair_list)
            if gt_mask is None:
                continue
            evaluator.accumulate(scoremap_abs, '', gt_mask)


def _get_evaluator(tester):
    metadata = configure_metadata(ospj(tester.args.metadata_root, 'test'))
    threshold_list = get_threshold_list(
        tester.args.cam_curve_interval,
        tester.args.threshold_type)
    evaluator = MaskEvaluator(metadata=metadata,
                              dataset_name=tester.args.dataset_name,
                              split='test',
                              threshold_list=threshold_list,
                              mask_root='',
                              iou_thresholds=None)
    return evaluator


def _get_data_loader(tester, superclass_labels):
    data_loader = get_data_loader(
        data_roots=tester.args.data_paths,
        metadata_root=tester.args.metadata_root,
        batch_size=tester.args.batch_size,
        workers=tester.args.workers,
        resize_size=tester.args.resize_size,
        crop_size=tester.args.crop_size,
        proxy_training_set=tester.args.proxy_training_set,
        superclass_labels=superclass_labels,
    )['test']
    return data_loader


def main():
    tester = Trainer()
    tester.model.eval()
    if tester.args.dataset_name != 'CUB':
        raise ValueError(
            "Evaluation of cue location is only possible on CUB dataset.")

    architecture = tester.args.architecture
    process = tester.args.score_map_process
    if tester.args.attribution_method == 'CALM_EM':
        process += '_EM'
    elif tester.args.attribution_method == 'CALM_ML':
        process += '_ML'

    data_path = ospj(tester.args.data_root, tester.args.dataset_name)
    trait = CUBTrait(data_path)
    evaluator = _get_evaluator(tester)

    pxap = {i: [] for i in NUM_DIFF_LIST}
    for num_diff in NUM_DIFF_LIST:
        print(f"Start class pairs with the number of "
              f"different parts to be {num_diff}")
        for class_pair_list in trait.diff2class_pair_dict[num_diff]:
            class_pair = [class_id - 1 for class_id in class_pair_list]
            data_loader = _get_data_loader(tester, class_pair)
            _cub_eval_inner_loop(tester, data_loader, class_pair, trait,
                                 class_pair_list, evaluator)
            pxap[num_diff].append(evaluator.compute()[0])
    mpxap = [float(f'{np.mean(pxap[i]):.02f}') for i in NUM_DIFF_LIST]
    print(f'\n{architecture}, {process} : {mpxap}\n')
    tester.logger.report(msg_dict={'step': 0,
                                   'test/cue_1': mpxap[0],
                                   'test/cue_2': mpxap[1],
                                   'test/cue_3': mpxap[2],
                                   })
    tester.logger.finalize_log()


if __name__ == '__main__':
    main()
