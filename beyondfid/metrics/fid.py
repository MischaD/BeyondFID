
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from beyondfid.log import logger

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from beyondfid.metrics import save_metric, register_metric, BaseMetric

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


@register_metric(name="fid")
class FID(BaseMetric):
    def __init__(self, config):
        super().__init__(config)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        mean_term = diff.dot(diff)
        var_term = np.trace(sigma1) + np.trace(sigma2) 
        cov_term = - 2 * tr_covmean
        fid = mean_term + var_term + cov_term
        return fid, mean_term, var_term, cov_term

    def calculate_activation_statistics(
        self, act 
    ):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def _compute(self, real, snth):
        try: 
            m1, s1 = self.calculate_activation_statistics(real.numpy())
            m2, s2 = self.calculate_activation_statistics(snth.numpy())
            fid, mean_term, var_term, cov_term = self.calculate_frechet_distance(m1, s1, m2, s2)
            return {"fid":float(fid), "mean":float(mean_term), "var":float(var_term), "cov":float(cov_term)}
        except ValueError as e: 
            logger.warning(f"FID computation failed due to {e}")
            logger.warning("Setting FID to -1")
            return {"fid":-1}


    def compute(self, train_features, test_features, snth_features):
        fid_train = self._compute(train_features, snth_features)
        fid_test = self._compute(test_features, snth_features)
        return {"fid_train": fid_train, "fid_test": fid_test}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        results = {}
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            fid = self.compute(train, test, snth)
            results[model] = fid

            if results_path is not None: 
                for key, value in fid.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=f"{key}", value=value)
        return results
