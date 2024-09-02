import torch.nn.functional as F
import numpy as np
from beyondfid.feature_extractor_models.inception import fid_inception_v3
from beyondfid.metrics import path_to_tensor
from scipy.stats import entropy
from beyondfid.metrics import save_metric
import os
import torch



def compute_is_score(config, output_path, results_path, hashtrain, hashtest, hashsnth):
    # uses pytorch conversion of inceptionv3 which has 1008 outputs related to this issue https://github.com/mseitzer/pytorch-fid/issues/43
    # not recommended as evaluation metrics see 
    inception = fid_inception_v3()
    inception.eval()
    def get_pred(x):
        x = inception.fc(x)
        return F.softmax(x, dim=0).data.cpu().numpy()

    splits = config.metrics.is_score.splits

    train, test, snth = path_to_tensor(output_path, "inception", hashtrain, hashtest, hashsnth)
    for ds, name in zip([train, test, snth], ["train", "test", "synth"]):
        # Get predictions
        #preds = np.zeros((len(ds), inception.fc.out_features))
        #batch_size = 1024

        #for i in np.arange(0, len(ds), step=batch_size):
        #    batch = ds[i*batch_size:(i+1)*batch_size]
        #    batch_size_i = len(batch) 

        #    preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

        ## Now compute the mean kl-div
        #split_scores = []

        #for k in range(splits):
        #    part = preds[k * (len(ds) // splits): (k+1) * (len(ds) // splits), :]
        #    py = np.mean(part, axis=0)
        #    scores = []
        #    for i in range(part.shape[0]):
        #        pyx = part[i, :]
        #        scores.append(entropy(pyx, py))
        #    split_scores.append(np.exp(np.mean(scores)))

        #mean, std = np.mean(split_scores), np.std(split_scores)

        # taken from: https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/image/inception.py

        idx = torch.randperm(ds.shape[0])
        features = ds[idx]

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

        save_metric(os.path.join(output_path, results_path), model="inception", key=f"is_mean_{name}", value=mean)
        save_metric(os.path.join(output_path, results_path), model="inception", key=f"is_std_{name}", value=std)

