import torch

__all__ = ['var_grad']


def var_grad(model, images, targets, num_classes, method,
             smooth_grad_nr_iter=50, smooth_grad_sigma=4., **kwargs):
    """
    Args:
        smooth_grad_nr_iter (int): Amount of images used to smooth gradient
        smooth_grad_sigma (int): Sigma multiplier when calculating std of noise
    """
    sum_of_square_saliency_map = torch.zeros(
        (images.size(0), images.size(2), images.size(3))).cuda()

    sum_of_saliency_map = torch.zeros(
        (images.size(0), images.size(2), images.size(3))).cuda()

    feasible_range = torch.max(images.view(images.size(0), -1), dim=1)[0] - \
                     torch.min(images.view(images.size(0), -1), dim=1)[0]
    sigma = feasible_range / smooth_grad_sigma

    for _ in range(smooth_grad_nr_iter):
        noise = torch.randn_like(images) * sigma.view((sigma.size(0), 1, 1, 1))
        noisy_images = images + noise
        saliency_map = method(model, noisy_images, targets, num_classes)
        sum_of_square_saliency_map += torch.mul(saliency_map, saliency_map)
        sum_of_saliency_map += saliency_map

    sum_of_square_saliency_map /= smooth_grad_nr_iter
    sum_of_saliency_map /= smooth_grad_nr_iter

    saliency_maps = sum_of_square_saliency_map - torch.mul(sum_of_saliency_map,
                                                           sum_of_saliency_map)

    return saliency_maps
