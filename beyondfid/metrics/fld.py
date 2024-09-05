import os
import math
from beyondfid.metrics import save_metric, register_metric, BaseMetric
from beyondfid.log import logger


@register_metric(name="fld")
class FLDMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
    
    def compute(self, train, test, snth):
        from fld.metrics.FLD import FLD
        test_fld = FLD(eval_feat="test").compute_metric(train, test, snth)
        train_fld = FLD(eval_feat="train").compute_metric(train, test, snth)

        if math.isnan(train_fld):
            logger.warning(f"Train FLD is NaN - setting it to -1")
            train_fld = -1
        if math.isnan(test_fld):
            logger.warning(f"Test FLD is NaN - setting it to -1")
            test_fld = -1

        return {"fld_train": train_fld, "fld_test": test_fld}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            metrics = self.compute(train, test, snth)
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)
