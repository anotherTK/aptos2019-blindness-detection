
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = F.hflip(image)
        return image


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = F.vflip(image)
        return image

class RandomRotation(object):
    def __init__(self, degrees, prob=0.5):
        self.t = torchvision.transforms.RandomRotation(degrees)
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.t(image)
        return image


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 prob=0.1
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = self.color_jitter(image)
        return image


class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image

class RandomResizedCrop(object):
    def __init__(self, size):
        self.t = torchvision.transforms.RandomResizedCrop(size)

    def __call__(self, image):
        return self.t(image)
