"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch
import torch.nn.functional as F


def add_subclass_maps(cams, labels, superclass, mean=False, minmax=False):
    output = torch.zeros([cams.size(0), cams.size(2), cams.size(3)],
                         device=torch.device('cuda:0'))
    for i, (label, cam) in enumerate(zip(labels, cams)):
        subclass_labels = superclass[label.item()]
        for subclass_label in subclass_labels:
            score_map = cam[subclass_label]
            if mean:
                score_map = score_map / len(subclass_labels)
            elif minmax:
                score_map -= score_map.min()
                score_map /= score_map.max()
            else:
                score_map = cam[subclass_label]
            output[i] += score_map
    return output


def cam_saliency(cams):
    output = torch.zeros([cams.size(0), cams.size(2), cams.size(3)],
                         device=torch.device('cuda:0'))
    for i, cam in enumerate(cams):
        for score_map in cam:
            score_map -= score_map.min()
            score_map /= score_map.max()
            output[i] += score_map
    return output


class UniversalProcess(object):
    def __init__(self, attribution_method):
        self.attribution_method = attribution_method

        self.conv_last = None
        self.attention = None

    def set_layers(self, conv_last, attention):
        self.conv_last = conv_last
        self.attention = attention

    def get_cam(self, f, f_former, labels,
                superclass=None, return_cam='vanilla'):
        batch_size = labels.size(0)
        if return_cam == 'vanilla':
            cam = f[torch.arange(batch_size), labels]
        elif return_cam == 'vanilla-saliency':
            cam = cam_saliency(f)
        elif return_cam == 'vanilla-superclass':
            cam = add_subclass_maps(f, labels, superclass, minmax=True)
        elif return_cam == 'saliency':
            cam = self._latent_prior(f, f_former).squeeze(1)
        elif return_cam == 'gtcond':
            cam = self._latent_posterior(f, f_former)
            cam = cam[torch.arange(batch_size), labels]
        elif return_cam == 'gtcond-superclass':
            cam = self._latent_posterior(f, f_former)
            cam = add_subclass_maps(cam, labels, superclass)
        elif return_cam == 'gtcond-superclass-mean':
            cam = self._latent_posterior(f, f_former)
            cam = add_subclass_maps(cam, labels, superclass, mean=True)
        elif return_cam == 'jointll':
            cam = self._joint_likelihood(f, f_former)
            cam = cam[torch.arange(batch_size), labels]
        elif return_cam == 'jointll-superclass':
            cam = self._joint_likelihood(f, f_former)
            cam = add_subclass_maps(cam, labels, superclass)
        elif return_cam == 'jointll-superclass-mean':
            cam = self._joint_likelihood(f, f_former)
            cam = add_subclass_maps(cam, labels, superclass, mean=True)
        else:
            raise ValueError(
                f'Invalid score_map_process argument: {return_cam}')
        return cam

    def input_for_loss(self, f, f_former):
        probs = None
        logits = None
        if self.attribution_method == 'CALM-EM':
            latent_posterior = self._latent_posterior(f, f_former)
            latent_posterior = latent_posterior.detach()
            joint_likelihood = self._joint_likelihood(f, f_former,
                                                      is_log=True)
            inputs = latent_posterior * joint_likelihood
            inputs = inputs.sum(dim=[2, 3])
            probs, _ = self._likelihood(f, f_former)
            probs = probs.view(probs.size(0), -1)
        else:
            inputs, logits = self._likelihood(f, f_former, is_log=True)
            inputs = inputs.view(inputs.size(0), -1)
            logits = logits.view(logits.size(0), -1)
            if self.attribution_method == 'CALM-ML':
                probs, _ = self._likelihood(f, f_former, is_log=False)
                probs = probs.view(probs.size(0), -1)
            elif self.attribution_method == 'CAM':
                probs = F.softmax(logits, dim=1)
            else:
                raise ValueError('Invalid attribution_method argument: '
                                 f'{self.attribution_method}')

        return {'logits': logits, 'probs': probs, 
                'inputs': inputs}

    def _likelihood(self, f, f_former, is_log=False):
        """ p(y|x) """
        if 'CALM' in self.attribution_method:
            joint_likelihood = self._joint_likelihood(f, f_former)
            likelihood = joint_likelihood.sum(dim=[2, 3], keepdim=True)
            likelihood = likelihood.log() if is_log else likelihood
        elif self.attribution_method == 'CAM':
            f = F.adaptive_avg_pool2d(f, (1, 1))
            likelihood = F.log_softmax(f, dim=1) if is_log \
                         else F.softmax(f, dim=1)
        else:
            raise ValueError('Invalid attribution_method argument: '
                             f'{self.attribution_method}')
        return likelihood, f

    def _latent_prior(self, f, f_former, is_log=False):
        """ p(z|x) """
        z = self.attention(f_former)
        batch_size, _, height, width = z.size()
        z = z / z.sum(dim=[2, 3], keepdim=True).detach()
        if is_log:
            z[z < 1e-8] += 1e-8
            z = z.log()
        return z

    def _latent_posterior(self, f, f_former):
        """ p(z|x,y) """
        numerator = self._joint_likelihood(f, f_former)
        denominator, _ = self._likelihood(f, f_former)
        latent_posterior = numerator / (denominator + 1e-10)
        return latent_posterior

    def _conditional_likelihood(self, f, is_log=False):
        """ p(y|x,z) """
        conditional_likelihood = F.log_softmax(f, dim=1) if is_log \
            else F.softmax(f, dim=1)
        return conditional_likelihood

    def _joint_likelihood(self, f, f_former, is_log=False):
        """  p(y,z|x) """
        conditional_likelihood = self._conditional_likelihood(f, is_log)
        latent_prior = self._latent_prior(None, f_former, is_log)
        joint_likelihood = conditional_likelihood + latent_prior \
            if is_log else \
            conditional_likelihood * latent_prior
        return joint_likelihood

