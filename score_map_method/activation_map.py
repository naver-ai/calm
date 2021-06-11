"""
CALM
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

__all__ = ['activation_map']


def activation_map(model, images, targets, score_map_process, 
                   superclass=None, **kwargs):
    cams = model(images, targets, superclass, return_cam=score_map_process)
    return cams
