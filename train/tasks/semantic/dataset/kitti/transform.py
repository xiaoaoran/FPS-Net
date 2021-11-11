# encoding=utf-8
"""
This manuscript defines augmentation processes for 3d point cloud scan.
Written by Xiao Aoran. aoran001@e.ntu.edu.sg
2020/Mar/02, 10:42.
"""

import random
import math

# axis of SemanticKITTI:
# X: forward
# Y: left
# Z: up

# random horizontal flip
class RandomLeftRightFlip(object):
    def __init__(self, p=0.5):
        """
        flip points in left-right direction, remain z direction
        :param p: probability to flip
        """
        self.p = p

    def __call__(self, points):
        """
        :param points: points to be fliped
        :return: flipped points
        """
        # points[:, 1] = - points[:, 1]
        if random.random() < self.p:
            points[:, 1] = - points[:, 1]  # y->-y, x and z remain
        return points

class RandomForwardBackwardFlip(object):
    def __init__(self, p=0.5):
        """
        flip points in forward-backward direction, remain z direction
        :param p: probability to flip
        """
        self.p = p

    def __call__(self, points):
        """
        :param points: points to be fliped
        :return: flipped points
        """
        # points[:, 1] = - points[:, 1]
        if random.random() < self.p:
            points[:, 0] = - points[:, 0]  # y->-y, x and z remain
        return points

class RandomRotation(object):
    def __init__(self, degree=1):
        """
        Rotate point cloud with angle degree. Default means randomly rotate 1°
        :param degree:
        """
        self.degree = degree / 360 * math.pi

    def __call__(self, points):
        degree = random.uniform(-1., 1.) * self.degree / 180 * math.pi
        x = points[:, 0]*math.cos(degree) - points[:, 1]*math.sin(degree)
        y = points[:, 1]*math.cos(degree) + points[:, 0]*math.sin(degree)
        points[:, 0] = x
        points[:, 1] = y
        return points


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
