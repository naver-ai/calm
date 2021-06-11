import torch

__all__ = ['backprop']


def backprop(model, images, targets, num_classes, **kwargs):
    images.requires_grad_()
    targets = targets.cuda()
    logits = model(images, targets)['logits']
    gradient = torch.eye(num_classes)[targets].cuda()
    logits.backward(gradient=gradient)
    saliency_maps, _ = torch.max(images.grad.data.abs(), dim=1)
    return saliency_maps
