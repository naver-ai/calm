"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import pickle
import torch

from util import get_scoremaps
from util import get_baseline
from util import resize_scoremaps
from util import get_topk_to_zero_mask
from util import random_mask_generator
from main import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

_ERASE_K_LIST = [1e-1, 3e-1, 5e-1, 7e-1, 9e-1]

def _get_correctness(tester, inputs, targets):
    with torch.no_grad():
        outputs = tester.model(inputs)['probs']
        preds = outputs.max(dim=1)[1].cpu().eq(targets).tolist()
    return preds


def main():
    tester = Trainer()
    tester.model.eval()

    device = tester.device
    dataset_name = tester.args.dataset_name
    data_loader = tester.loaders['test']

    measures = {'preds': [],
                'preds-topk': {erase_k: [] for erase_k in _ERASE_K_LIST},
                'preds-random': {erase_k: [] for erase_k in _ERASE_K_LIST}}

    process = tester.args.score_map_process
    if tester.args.attribution_method == 'CALM_EM':
        process += '_EM'
    elif tester.args.attribution_method == 'CALM_ML':
        process += '_ML'

    print(f'start! {process}')

    for enum, (inputs, targets, _) in enumerate(data_loader):
        inputs = inputs.to(device)
        baselines = get_baseline('zero', inputs, device)
        preds = _get_correctness(tester, inputs, targets)
        measures['preds'] += preds

        scoremaps = get_scoremaps(tester, inputs, targets, dataset_name)
        scoremaps = resize_scoremaps(scoremaps).to(device)

        for erase_method in ['topk', 'random']:
            for erase_idx, erase_k in enumerate(_ERASE_K_LIST):
                if erase_method == 'topk':
                    masks = get_topk_to_zero_mask(scoremaps, erase_k, device)
                elif erase_method == 'random':
                    masks = random_mask_generator(inputs, device, erase_k)
                else:
                    raise ValueError('Error!')
                inputs_masked = inputs * masks + baselines * (1 - masks)
                preds_masked = _get_correctness(tester, inputs_masked, targets)
                measures[f'preds-{erase_method}'][erase_k] += preds_masked

        if (enum + 1) % 50 == 0:
            print(f'Iteration: {enum + 1}')

    with open(f'{tester.args.log_folder}/measure_{process}.pickle', 'wb') as f:
        pickle.dump(measures, f)

    for k in _ERASE_K_LIST:
        ratio = sum(measures['preds-topk'][k]) / sum(
            measures['preds-random'][k])
        print(f'erase_k: {k}, prediction ratio w.r.t. random erase: '
              f'{ratio:.04f}')


if __name__ == '__main__':
    main()
