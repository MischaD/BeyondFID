import inspect
import sys
from abc import ABC, abstractmethod
from torch import nn


_FEATURE_MODELS = {}


class BaseFeatureModel(ABC):
    @abstractmethod
    def compute_latent(self):
        pass

    def __call__(self, *args, **kwds):
        return self.compute_latent(*args, **kwds)


def register_feature_model(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _FEATURE_MODELS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _FEATURE_MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


@register_feature_model(name="generic")
class GenericFeatureModel(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self._compute_latent = model_config.forward

    def compute_latent(self, x):
        return self._compute_latent(x)


def load_feature_model(config):
    model_name = config.name
    if model_name not in _FEATURE_MODELS:
        raise ValueError(f"No model found with name {model_name}")
    
    model_class = _FEATURE_MODELS[model_name]
    return model_class(config.config)


# register modules
from beyondfid.feature_extractor_models.byol import BYOL
from beyondfid.feature_extractor_models.inception import InceptionV3
from beyondfid.feature_extractor_models.dinov2 import DINOv2
from beyondfid.feature_extractor_models.clip import CLIP 
from beyondfid.feature_extractor_models.convnext import ConvNeXTEncoder 
from beyondfid.feature_extractor_models.data2vec import HuggingFaceTransformerEncoder
from beyondfid.feature_extractor_models.swav import ResNet50Encoder
