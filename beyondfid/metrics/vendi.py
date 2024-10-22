# taken from https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/metrics/vendi.py
import os
import numpy as np
import torch
from beyondfid.log import logger
from beyondfid.metrics import save_metric, register_metric, BaseMetric
import sklearn.metrics
import torch
from sklearn import preprocessing
from sklearn.metrics.pairwise import polynomial_kernel
import scipy
import scipy.linalg
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics.pairwise import polynomial_kernel
import scipy
import scipy.linalg
import torch
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize
from torch.linalg import eigvalsh
from beyondfid.default_config import config


@register_metric(name="vendi")
class VendiMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    
    def compute(self, train_features, test_features, snth_features):
        """
        Computes precision, recall, density, and coverage given two manifolds.

        Args:
            train_features: torch.tensor([N, feature_dim], dtype=torch.float32)
            test_features: torch.tensor([N, feature_dim], dtype=torch.float32)
            snth_features: torch.tensor([N, feature_dim], dtype=torch.float32)
        Returns:
            dict of precision, recall, density, and coverage.
        """
        train = self._compute(train_features)
        test = self._compute(test_features)
        snth = self._compute(snth_features)
        return {"vendi_train": train, 
                "vendi_test": test,
                "vendi_snth": snth}


    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        results = {}
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)

            vendis = self.compute(train, test, snth)
            results[model] = vendis

            if results_path is not None: 
                for key, value in vendis.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=f"{key}", value=value)
        return results


    def _compute(self, X):
        if len(X) > self.config.max_size:
            print(f"Computing vendi score can be really expensive for large datasets.\n Subsampling {self.config.max_size} images. Set config.metrics.vendi.max_size higher if wanted")
            shuffled_indices = torch.randperm(len(X))[:self.config.max_size]
            X = X[shuffled_indices]

        # Move X to GPU and convert it to a PyTorch tensor
        X = torch.tensor(X, device='cuda', dtype=torch.float32)

        if self.config.normalize:
            X = F.normalize(X, p=2, dim=1)  # Use F.normalize for PyTorch tensor

        n = X.shape[0]
        
        if self.config.kernel == 'linear':
            S = X @ X.T  # Matrix multiplication on GPU
        else:
            raise NotImplementedError("kernel not implemented")
        
        X = X.cpu()

        # Compute eigenvalues using torch.linalg.eigvalsh on GPU
        w = eigvalsh(S / n)
        
        return float(np.exp(self.entropy_q(w.cpu(), q=self.config.q)))


    def entropy_q(self, p, q=1):
        p_ = p[p > 0]
        if q == 1:
            return -(p_ * np.log(p_)).sum()
        if q == "inf":
            return -np.log(np.max(p))
        return np.log((p_ ** q).sum()) / (1 - q)
