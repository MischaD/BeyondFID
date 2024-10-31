import torch.nn.functional as F
import numpy as np
from beyondfid.feature_extractor_models.inception import fid_inception_v3
from scipy.stats import entropy
from beyondfid.metrics import save_metric, register_metric, BaseMetric
import os
import torch

@register_metric(name="is_score")
class ISScore(BaseMetric):
    def __init__(self, config):
        super().__init__(config)

        self.inception = fid_inception_v3()
        self.inception.eval()

    @property
    def models(self): 
        return "inception" 
    
    def set_models(self, new_models: list):
        raise ValueError("Inceptions score only implemented for Inception")

    def get_inception_pred(self,x):
        x = self.inception.fc(x)
        return x.cpu() # F.softmax(x, dim=0).data.cpu()

    def compute(self, train_features, test_features, snth_features):
        """
        """
        splits = self.config.splits

        results = {}
        for ds, name in zip([train_features, test_features, snth_features], ["train", "test", "synth"]):
            # Get predictions
            preds = torch.zeros((len(ds), self.inception.fc.out_features))
            batch_size = 1024

            for i in torch.arange(0, len(ds), step=batch_size):
                batch = ds[i*batch_size:(i+1)*batch_size]
                batch_size_i = len(batch) 

                preds[i*batch_size:i*batch_size + batch_size_i] = self.get_inception_pred(batch)

            # taken from: https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/inception.py

            idx = torch.randperm(ds.shape[0])
            #preds = preds.transpose(0, 1)
            features = preds[idx]

            # calculate probs and logits
            prob = features.softmax(dim=1)
            log_prob = features.log_softmax(dim=1)

            # split into groups
            prob = prob.chunk(splits, dim=0)
            log_prob = log_prob.chunk(splits, dim=0)

            # calculate score per split
            mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
            kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
            kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
            kl = torch.stack(kl_)

            # return mean and std
            mean = float(kl.mean())
            std = float(kl.std())
            results[f"is_score_{name}"] = {"mean":mean, "std":std}

        return results    

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        # uses pytorch conversion of inceptionv3 which has 1008 outputs related to this issue https://github.com/mseitzer/pytorch-fid/issues/43
        # not recommended as evaluation metrics see 
        train, test, snth = self.path_to_tensor(output_path, "inception", hashtrain, hashtest, hashsnth)

        is_score = self.compute(train, test, snth)
        if results_path is not None: 
            save_metric(os.path.join(output_path, results_path), model="inception", key=f"is_score", value=is_score)
        return is_score

