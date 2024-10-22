import os
import numpy as np
from beyondfid.metrics import save_metric, register_metric, BaseMetric
from tqdm import tqdm
import torch


@register_metric(name="diversity")
class DiversityMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.alphas = config.alphas
        self.batch_size = config.batch_size
        self.folds = config.folds
    
    def get_n_train(self, features, alpha):
        n_total = len(features)
        n_train = int(n_total//(1+alpha))
        n_test = n_total - n_train
        return n_total, n_train, n_test

    def compute_support(self, features_a, features_b):
        features_train = features_a.to("cuda") # real / training data
        features_test = features_b # test or synthetic

        closest = []
        for i in tqdm(range(0, features_test.size(0), self.batch_size), desc="Processing Batches"):
            # 512 new 'generated' images each batch 
            batch_features = features_test[i:i+self.batch_size].to("cuda")

            dist = torch.cdist(features_train, batch_features, p=2)
            dist = dist.argmin(dim=0).cpu()
            batch_features.cpu()
            closest.extend(dist.tolist())

        features_train.cpu()
        dist.size()
        perc = len(set(closest)) / len(features_train)
        return closest, perc

    def compute_closest_for_alpha(self, features, alpha, fold=0):
        """ alpha == N_test / N_train """
        n_total, n_train, n_test = self.get_n_train(features, alpha)

        file_nums = np.arange(n_total)
        np.random.seed(fold)
        np.random.shuffle(file_nums)

        features_train = features[file_nums[:n_train]].to("cuda")
        features_test = features[file_nums[n_train:]]

        closest, perc = self.compute_support(features_train, features_test)
        return closest, perc

    def compute_train_only(self, train): 
        results = {}
        for alpha in self.alphas: 
            percs = []
            for fold in range(self.folds):
                closest, perc = self.compute_closest_for_alpha(train, alpha, fold=fold)
                percs.append(perc)
            results[alpha] = {"mean": float(np.array(percs).mean()), "std": float(np.array(percs).std())}
        return {"diversity_train_only": results}

    def compute(self, train, test, snth): 
        # fold not supported right now. 
        return {"diversity_train": self.compute_support(train, snth)[1], 
                "diversity_test": self.compute_support(test, snth)[1],
                }

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        results = {}
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            if hashtrain == hashsnth: 
                metrics = self.compute_train_only(train)
            else: 
                metrics = self.compute_with_synth(train, test, snth)

            results[model] = metrics
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)
        return results

