# adapted from https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/models/data2vec.py
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model


@register_feature_model(name="data2vec")
class HuggingFaceTransformerEncoder(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        self.model_config = model_config
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_config.checkpoint)

        self.model = AutoModel.from_pretrained(self.model_config.checkpoint, add_pooling_layer=True)

    def transform(self, image):
        return self.image_processor(image, return_tensors="pt")

    def compute_latent(self, x): 
        x = self.transform(x)
        f = self.model.forward(x).pooler_output
        return f