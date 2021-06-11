import torch
import numpy as np


__all__ = ['integrated_grad']


def generate_images_on_linear_path(images, nr_iter):
    step_array = np.arange(nr_iter + 1) / nr_iter
    xbar_list = [images * step for step in step_array]
    return xbar_list


def integrated_grad(model, images, targets, num_classes, method,
                    integrated_grad_nr_iter, **kwargs):
    xbar_list = generate_images_on_linear_path(images, integrated_grad_nr_iter)
    saliency_maps = torch.zeros(
        (images.size(0), images.size(2), images.size(3))).cuda()
    for xbar_images in xbar_list:
        single_integrated_grad = \
            method(model, xbar_images, targets, num_classes)
        saliency_maps = saliency_maps + single_integrated_grad
    saliency_maps /= integrated_grad_nr_iter

    return saliency_maps
