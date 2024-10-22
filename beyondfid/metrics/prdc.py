
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
import torch


@register_metric(name="prdc")
class PRDCMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.device = torch.device("cuda:0")
    
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
        # only compute prdc on subset 

        
        prdc = self._compute(train_features, snth_features)
        return prdc


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

    def compute_pairwise_distance(self, data_x, data_y=None, batch_size=1000):
        """
        Compute pairwise Euclidean distance between two sets of data using PyTorch tensors in batches.
        The computation is done on GPU if available.

        Args:
            data_x: torch.Tensor([N, feature_dim], dtype=torch.float32)
            data_y: torch.Tensor([M, feature_dim], dtype=torch.float32)
            batch_size: int, the size of the sub-batches to avoid memory overload.
        Returns:
            Pairwise distances matrix: torch.Tensor([N, M], dtype=torch.float32)
        """
        if data_y is None:
            data_y = data_x

        # Ensure data is on the correct device (GPU if available)
        data_x = data_x.to(self.config.device)
        data_y = data_y.to(self.config.device)

        n_samples_x = data_x.shape[0]
        n_samples_y = data_y.shape[0]

        # Initialize the result matrix to store the distances
        dists = torch.zeros((n_samples_x, n_samples_y), dtype=torch.float32)

        # Process the data in batches to avoid memory overload
        for i in range(0, n_samples_x, batch_size):
            end_i = min(i + batch_size, n_samples_x)
            batch_x = data_x[i:end_i]

            # Compute pairwise distances for the current batch
            dists[i:end_i, :] = torch.cdist(batch_x, data_y, p=2).cpu()

        return dists

    def batch_compute_kth_value(self, data_x, data_y=None, k=1, batch_size=1000):
        """
        Compute k-th nearest neighbor distances in batches on GPU.
        
        Args:
            data_x: torch.Tensor([N, feature_dim], dtype=torch.float32)
            data_y: torch.Tensor([M, feature_dim], dtype=torch.float32)
            k: int, which nearest neighbor to consider.
            batch_size: int, size of each batch for processing to avoid OOM issues.
        
        Returns:
            kth_distances: torch.Tensor([N], dtype=torch.float32)
        """
        if data_y is None:
            data_y = data_x

        n_samples_x = data_x.shape[0]
        data_y = data_y.to(self.config.device)

        # Initialize the array to store the k-th nearest distances
        kth_distances = torch.zeros(n_samples_x, dtype=torch.float32, device=self.config.device)

        # Process the data in batches
        for i in range(0, n_samples_x, batch_size):
            end_i = min(i + batch_size, n_samples_x)
            batch_x = data_x[i:end_i].to(self.config.device)

            # Compute pairwise distances between the current batch of data_x and data_y
            dists = torch.cdist(batch_x, data_y, p=2)

            # Find the k-th nearest distances for each sample in the batch
            kth_values_batch = torch.topk(dists, k=k + 1, largest=False).values[:, -1]
            
            # Store the k-th values for this batch
            kth_distances[i:end_i] = kth_values_batch
        
        data_y = data_y.cpu()
        return kth_distances.cpu()

    def compute_nearest_neighbour_distances(self, input_features, nearest_k, batch_size=1000):
        """
        Compute nearest neighbor distances in batches using GPU.
        
        Args:
            input_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            nearest_k: int, the k-th nearest neighbor.
            batch_size: int, size of each batch for processing.
        
        Returns:
            radii: torch.Tensor([N], dtype=torch.float32)
        """
        # Compute k-th nearest neighbor distances on-the-fly without saving a large distance matrix
        radii = self.batch_compute_kth_value(input_features, k=nearest_k, batch_size=batch_size)
        return radii

    def _compute(self, real_features, fake_features):
        """
        Computes precision, recall, density, and coverage given two manifolds.

        Args:
            real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            fake_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """

        max_len = min(len(real_features), len(fake_features))
        if max_len != max(len(real_features), len(fake_features)):
            print(f"Computing precision and recall should be done on two datasets of equal size. Subsampling the larger dataset.")
            print(f"Computing PRDC on a random subset of equal size: {max_len}.")
        if len(real_features) > len(fake_features): 
            shuffled_indices = torch.randperm(len(real_features))[:max_len]
            real_features = real_features[shuffled_indices]
        elif len(real_features) < len(fake_features): 
            shuffled_indices = torch.randperm(len(fake_features))[:max_len]
            fake_features = fake_features[shuffled_indices]

        distance_real_fake = self.compute_pairwise_distance(
            real_features, fake_features)

        real_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(
            real_features, self.config.nearest_k)
        fake_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(
            fake_features, self.config.nearest_k)

        precision = (
            distance_real_fake <
            real_nearest_neighbour_distances.unsqueeze(1)
        ).any(dim=0).float().mean()

        recall = (
            distance_real_fake <
            fake_nearest_neighbour_distances.unsqueeze(0)
        ).any(dim=1).float().mean()

        density = (1. / float(self.config.nearest_k)) * (
            distance_real_fake <
            real_nearest_neighbour_distances.unsqueeze(1)
        ).sum(dim=0).float().mean()

        coverage = (
            distance_real_fake.min(dim=1)[0] <
            real_nearest_neighbour_distances
        ).float().mean()

        return dict(precision=precision.item(), recall=recall.item(),
                    density=density.item(), coverage=coverage.item())