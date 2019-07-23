

from torchvision.models.resnet import ResNet, Bottleneck
from cls_bm.utils.model_serialization import trim_model, align_and_update_state_dicts

from torch.utils import model_zoo

dependencies = ['torch', 'torchvision']


model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}


def _resnext(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    
    loaded_state_dict = model_zoo.load_url(model_urls[arch])
    loaded_state_dict = trim_model(loaded_state_dict, ["fc"])
    model_state_dict = model.state_dict()
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def resnext101_32x8d_wsl(**kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, **kwargs)


def resnext101_32x16d_wsl(**kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, **kwargs)


def resnext101_32x32d_wsl(**kwargs):
    """Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32
    return _resnext('resnext101_32x32d', Bottleneck, [3, 4, 23, 3], True, **kwargs)


def resnext101_32x48d_wsl(**kwargs):
    """Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48
    return _resnext('resnext101_32x48d', Bottleneck, [3, 4, 23, 3], True, **kwargs)
