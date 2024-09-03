import torch
from torch import nn
import torchvision.transforms as transforms
from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model


@register_feature_model(name="dinov2")
class DINOv2(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

        # From https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L44
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)#, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.model.eval()
        self.model.to("cuda")

    def compute_latent(self, x):
        x = self.preprocess(x)
        with torch.no_grad():
            y = self.model(x)
        return y