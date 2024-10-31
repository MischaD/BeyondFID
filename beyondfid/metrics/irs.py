import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.optimize import fsolve
import numpy as np
from functools import lru_cache
from beyondfid.metrics import save_metric, register_metric, BaseMetric
import numpy as np
from tqdm import tqdm
from beyondfid.log import logger
import torch
import os


class LogFactorial:
    def __init__(self, max_value):
        """Initialize the LogFactorial class with a precomputed table up to max_value."""
        self.max_value = max_value
        self.logfactorial_lookup = self.compute_logfactorial_lookup(max_value)

    def compute_logfactorial_lookup(self, max_value):
        """Compute the logfactorial values up to max_value and store them in a list."""
        logfactorial_lookup = [0] * (max_value + 1)  # Initialize the list with zeros
        for i in range(2, max_value + 1):
            logfactorial_lookup[i] = logfactorial_lookup[i - 1] + np.log(i)
        return logfactorial_lookup

    def extend_lookup(self, new_max_value):
        """Extend the lookup table if the requested value exceeds the current max_value."""
        # Only extend if new_max_value is larger than current max_value
        if new_max_value > self.max_value:
            logger.info(f"Extending LogFactorial lookup table to {new_max_value}")
            # Extend the current table to new_max_value
            for i in range(self.max_value + 1, new_max_value + 1):
                self.logfactorial_lookup.append(self.logfactorial_lookup[i - 1] + np.log(i))
            self.max_value = new_max_value

    def __call__(self, x):
        """Make the class callable to return log(x!)."""
        if x > self.max_value:
            # If x exceeds the current max_value, extend the lookup table
            self.extend_lookup(x)
        return self.logfactorial_lookup[x]


def log_binom(n, k): 
    return logfactorial(n) - logfactorial(k) - logfactorial(n-k)


# Stirling number approximation function
def log_stirling_second_kind_approx(n, k):
    # Compute v and solve for G
    if n == k: 
        return 0 
    assert k > 0  and k < n 

    v = n / k

    # Define the function for G
    def G_func(G):
        return G - v * np.exp(G - v)

    # Solve for G
    G_initial_guess = 0.5  # Starting guess for G
    G = fsolve(G_func, G_initial_guess)[0]

    #print(f"n {n} -- k {k} -- v {v} -- G {G}")

    # Compute the other parts of the approximation formula
    part1 = 0.5 * np.log((v - 1) / (v * (1 - G)))
    part2 = (n-k) * np.log(((v - 1) / (v - G)))
    part3 = n*np.log(k) -  k * np.log(n) + k * (1 - G)

    # Combine parts with binomial coefficient
    approximation = part1 + part2 + part3 + log_binom(n, k)

    return approximation


def log_compute_formula(s, k, n): 
    logstir = log_stirling_second_kind_approx(n, k)
    return logstir + logfactorial(s) - logfactorial(s-k) - n * np.log(s)


logfactorial = LogFactorial(int(5e4)) # will be recomputed


@register_metric(name="irs")
class IRSMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.alpha_e = config.alpha_e
        self.confidence = True
        self.prob_tolerance = config.prob_tolerance
        self.naive = config.naive
        self.batch_size = config.batch_size # for computation of closes neighbour


    def compute_irs_inf(self, n_train_max, n_sampled, k_measured): 
        # n_train_max = len(train_dataset)
        # n_sampled = python sample.py --N=n_sampled --outpath=samples
        # k_measured = beyondfid irs --trainpath=train_dataset --synthpath=samples

        n_train = n_train_max
        alpha_e = self.alpha_e
        confidence= self.confidence 
        prob_tolerance= self.prob_tolerance
        naive= self.naive
        
        print(f"Maximum Possible Number of Train Images: {n_train_max}\nSampled images: {n_sampled}\nLearned images: {k_measured}")
        alpha_of_IRS_alpha = n_sampled / n_train
        print(f"IRS (alpha={alpha_of_IRS_alpha:.2f}): {k_measured / n_train_max}")
        
        if naive == True: 
            probs = []
            n_train_ests = [*range(k_measured, n_train_max)]
            for n_train_est in n_train_ests: 
                alpha_of_IRS_alpha = n_sampled / n_train_est 
                irs_alpha = np.exp(log_compute_formula(s=n_train_est, k=k_measured, n=n_sampled))
                probs.append(irs_alpha)
                if len(probs) > 2 and probs[-2] > probs[-1] and irs_alpha < prob_tolerance: 
                    break
            probs = np.array(probs)

            irs_inf = np.argmax(probs)
            n_learned_pred = n_train_ests[irs_inf] # most likely 

        else: 
            # do binary search instead of naive search. Uses the fact that ther is a single mode in the function over different s (note- not a distribution)
            low = k_measured
            high = n_train_max
            while low <= high:
                mid = (low + high) // 2

                # has to be either mid-1, mid, or mid+1
                if high - low == 2: 
                    break

                prob_mid_m1 = log_compute_formula(s=mid-1, k=k_measured, n=n_sampled)
                prob_mid = log_compute_formula(s=mid, k=k_measured, n=n_sampled)
                prob_mid_p1 = log_compute_formula(s=mid+1, k=k_measured, n=n_sampled)
                #print(f"Low: {low} -- mid: {mid} -- high: {high} ")
                #print(f"m1: {prob_mid_m1} -- mid: {prob_mid} -- mid+1: {prob_mid_p1}")

                if prob_mid > max(prob_mid_m1, prob_mid_p1): 
                    break # mid is highest  
                if prob_mid >= prob_mid_p1: 
                    high = mid
                else: 
                    low = mid

            prob_mid_l1 = log_compute_formula(s=mid-1, k=k_measured, n=n_sampled)
            prob_mid = log_compute_formula(s=mid, k=k_measured, n=n_sampled) 
            prob_mid_u1 = log_compute_formula(s=mid+1, k=k_measured, n=n_sampled)
            n_learned_pred = mid - 1 + np.argmax(np.array([prob_mid_l1, prob_mid, prob_mid_u1]))


        # cannot have more than kmax different images
        kmax = min(n_train, n_sampled)

        if confidence == True: 
            # Binary search for n_learned_high
            low, high = n_learned_pred + 1, n_train_max
            while low <= high:
                mid = (low + high) // 2
                prob = 0 
                prob_k = 1  # Just a dummy value

                for k in range(min(k_measured, n_sampled), 0, -1):  # reversed
                    if prob_k > prob_tolerance:
                        prob_k = np.exp(log_compute_formula(s=mid, k=k, n=n_sampled))
                        prob += prob_k
                
                if prob < alpha_e:
                    high = mid - 1  # Search the lower half
                else:
                    low = mid + 1  # Search the upper half
            n_learned_high = high  # The largest value that satisfies the condition

            # Binary search for n_learned_low
            low, high = k_measured, n_learned_pred - 1
            while low <= high:
                mid = (low + high) // 2
                prob = 0
                prob_k = 1  # Just a dummy value

                for k in range(k_measured, kmax + 1):
                    if prob_k > prob_tolerance:
                        prob_k = np.exp(log_compute_formula(s=mid, k=k, n=n_sampled))
                        prob += prob_k

                if prob < alpha_e:
                    low = mid + 1  # Search the upper half
                else:
                    high = mid - 1  # Search the lower half

            n_learned_low = low  # The smallest value that satisfies the condition

        irs_pred = n_learned_pred / n_train_max
        print(f"IRS (inf): {irs_pred}")
        print(f"Predicted number of images for IRS_infinity: {n_learned_pred} -- IRS: {irs_pred}")

        if confidence == True: 
            irs_prep_higher = n_learned_high / n_train_max
            irs_prep_lower = n_learned_low / n_train_max
            print(f"Predicted number of images for IRS_infinity,H: {n_learned_high} -- IRS: {irs_prep_higher}")
            print(f"Predicted number of images for IRS_infinity,L: {n_learned_low} -- IRS: {irs_prep_lower}\n")
            return (irs_prep_lower, irs_pred, irs_prep_higher)

        return (None, irs_pred, None)

    
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
        # test is ignored for now 
        n_train = len(train)
        n_ref_test = len(test)
        n_ref_snth = len(snth)
        n_ref = min(n_ref_test, n_ref_snth)

        # Define a generator with a fixed seed
        generator = torch.Generator()
        generator.manual_seed(42)  # Replace 42 with your desired seed value

        if n_ref_snth > n_ref: 
            logger.info(f"Randomly sampling {n_ref} synthetic images to make reference estimate IRS_a more accurate")
            rnd_idx = torch.randperm(n_ref_snth, generator=generator)[:n_ref]
            snth = snth[rnd_idx]
        elif n_ref_test > n_ref: 
            logger.info(f"Randomly sampling {n_ref} test images to make reference estimate IRS_a more accurate. Consider using multiple folds")
            rnd_idx = torch.randperm(n_ref_test, generator=generator)[:n_ref]
            test = test[rnd_idx]

        results = {}
        for name, ref_data in zip(["test", "snth"], [test, snth]): 
            closest, perc = self.compute_support(train, ref_data)

            k_measured = len(set(closest))
            alpha = k_measured / n_train

            logger.info(f"Computing IRS results")

            irs_pred_lower, irs_pred, irs_prep_higher = self.compute_irs_inf(n_train, len(ref_data), k_measured)
            k_learned_pred = int(irs_pred * n_train)

            results[name] = {"n_train": n_train, 
                "n_sampled": len(ref_data), 
                "k_measured": k_measured,
                "alpha": alpha,
                "irs_alpha": perc, 
                "irs_inf_u": irs_prep_higher,
                "irs_inf_l": irs_pred_lower,
                "irs_inf": irs_pred,
                "k_pred_inf": k_learned_pred,
            }
        
        results["irs_adjusted"] = results["snth"]["k_pred_inf"] / results["test"]["k_pred_inf"]
        return results


    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        results = {}
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            
            # test data is unused
            metrics = self.compute(train, test, snth)
            results[model] = metrics
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)
        return results

