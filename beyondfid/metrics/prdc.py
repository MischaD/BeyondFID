
"""
https://github.com/clovaai/generative-evaluation-prdc

prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
import numpy as np
import torch
from beyondfid.log import logger
from beyondfid.metrics import save_metric, register_metric, BaseMetric
import sklearn.metrics


@register_metric(name="prdc")
class PRDCMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)

    def compute_pairwise_distance(self, data_x, data_y=None):
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(
            data_x, data_y, metric='euclidean', n_jobs=8)
        return dists

    def get_kth_value(self, unsorted, k, axis=-1):
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def compute_nearest_neighbour_distances(self, input_features, nearest_k):
        distances = self.compute_pairwise_distance(input_features)
        radii = self.get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def _compute(self, real_features, fake_features):
        """
        Computes precision, recall, density, and coverage given two manifolds.

        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """
        real_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(
            real_features, self.config.nearest_k)
        fake_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(
            fake_features, self.config.nearest_k)
        distance_real_fake = self.compute_pairwise_distance(
            real_features, fake_features)

        precision = (
                distance_real_fake <
                np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).any(axis=0).mean()

        recall = (
                distance_real_fake <
                np.expand_dims(fake_nearest_neighbour_distances, axis=0)
        ).any(axis=1).mean()

        density = (1. / float(self.config.nearest_k)) * (
                distance_real_fake <
                np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).sum(axis=0).mean()

        coverage = (
                distance_real_fake.min(axis=1) <
                real_nearest_neighbour_distances
        ).mean()

        return dict(precision=precision, recall=recall,
                    density=density, coverage=coverage)
    
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
        train = self._compute(train_features, snth_features)
        test = self._compute(test_features, snth_features)
        return {"prdc_train": train, 
                "prdc_test": test}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        results = {}
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)

            prdc = self.compute(train, test, snth)
            results[model] = prdc

            if results_path is not None: 
                for key, value in prdc.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=f"{key}", value=value)
        return results