import torch

__all__ = ['input_grad']


def input_grad(model, images, targets, num_classes, method, **kwargs):
    return method(model, images, targets, num_classes) 
