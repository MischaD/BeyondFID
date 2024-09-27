# adapted from https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/models/data2vec.py
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from beyondfid.feature_extractor_models import BaseFeatureModel, register_feature_model


@register_feature_model(name="data2vec")
class HuggingFaceTransformerEncoder(BaseFeatureModel, nn.Module):
    def __init__(self, model_config):
        super(HuggingFaceTransformerEncoder, self).__init__()  # Properly initialize nn.Module
        self.model_config = model_config
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_config.checkpoint)

        self.model = AutoModel.from_pretrained(self.model_config.checkpoint, add_pooling_layer=True)

    def transform(self, image):
        return self.image_processor(image, return_tensors="pt")

    def compute_latent(self, x): 
        device = x.device
        x_ = self.transform(x).to(device)
        f = self.model(x_["pixel_values"],return_dict=True).pooler_output
        return f