import os
from beyondfid.metrics import save_metric, register_metric, BaseMetric
from beyondfid.log import logger


@register_metric(name="cttest")
class CTTestMetric(BaseMetric):
    def __init__(self, config):
        super().__init__(config)
    
    def compute(self, train, test, snth):
        from fld.metrics.CTTest import CTTest
        try:
            cttest = CTTest().compute_metric(train, test, snth)
        except Exception as e:
            logger.warning(f"cttest failed due to {e}\nSetting cttest to -1")
            cttest = -1
        return {"cttest": cttest}

    def compute_from_path(self, output_path, hashtrain, hashtest, hashsnth, results_path=None):
        for model in self.models:
            train, test, snth = self.path_to_tensor(output_path, model, hashtrain, hashtest, hashsnth)
            metrics = self.compute(train, test, snth)
            if results_path is not None: 
                for key, value in metrics.items():
                    save_metric(os.path.join(output_path, results_path), model=model, key=key, value=value)
