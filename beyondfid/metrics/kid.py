import os
from beyondfid.metrics import save_metric, register_metric, BaseMetric


@register_metric(name="kid")
class KIDMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
    
    def compute(self, train, test, snth):
        from fld.metrics.KID import KID
        test_kid = KID(ref_feat="test").compute_metric(None, test, snth)
        train_kid = KID(ref_feat="train").compute_metric(train, None, snth)

        return {"kid_train": train_kid, "kid_test": test_kid}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            metrics = self.compute(train, test, snth)
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)

