# adapted from https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/models/convnext.py
from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model
from torchvision import transforms
from torch import nn
from timm.models import create_model
from timm.data.constants import  IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import sys


@register_feature_model(name="convnext")
class ConvNeXTEncoder(BaseFeatureModel, nn.Module):
    """
    requires timm version: 0.8.19.dev0
    model_arch options: 
        convnext_xlarge_in22k (imagenet 21k); default
        convnext_xxlarge.clip_laion2b_rewind (clip objective trained on laion2b)

    see more options https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py

    """
    def __init__(self, model_config):
        super().__init__()
        self.arch = model_config.arch
        self.model = create_model(
                    self.arch,
                    pretrained=True,
                )
        self.model.eval()

        if self.arch == "convnext_xlarge.fb_in22k":
            self.input_size = 224
        elif self.arch in ["convnext_base.clip_laion2b_augreg", "convnext_xxlarge.clip_laion2b_rewind"]:
            self.input_size = 256

        self.build_transform()


    def build_transform(self):
        # get mean & std based on the model arch
        if self.arch == "convnext_xlarge.fb_in22k":
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        elif "clip" in self.arch:
            mean = OPENAI_CLIP_MEAN
            std = OPENAI_CLIP_STD

        t = []

        # warping (no cropping) when evaluated at 384 or larger
        if self.input_size >= 384:
            t.append(
            transforms.Resize((self.input_size, self.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {self.input_size} size input images...", file=sys.stderr)
        else:
            size = 256
            t.append(
                # to maintain same ratio
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(self.input_size))

        t.append(transforms.Normalize(mean, std))
        self.transform_ops = transforms.Compose(t)

    def transform(self, image):
        return self.transform_ops(image)

    def compute_latent(self, x):
        x = self.transform(x)

        # forward features + global_pool + norm + flatten => output dims ()
        outputs = self.model.forward_features(x)
        outputs = self.model.head.global_pool(outputs)
        outputs = self.model.head.norm(outputs)
        outputs = self.model.head.flatten(outputs)
        return outputs