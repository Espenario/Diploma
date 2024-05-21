import optuna
import numpy as np
from source.onn_model import OnnContourExtractionModule
from source.pipelines import OnnHyperPipeline
from source.utils import *
from source.datasets import HyperSpectralData

class ContExtrObjective:

    def __init__(self, 
                 min_level:float, 
                 max_level:float, 
                 img:np.ndarray,
                 cont_extr_module:OnnContourExtractionModule,
                 cont_area_threshold_percent_min:float = 1,
                 cont_area_threshold_percent_max:float = 10):

        self.min_level = min_level
        self.max_level = max_level
        self.cont_area_threshold_percent_min = cont_area_threshold_percent_min
        self.cont_area_threshold_percent_max = cont_area_threshold_percent_max
        self.cont_extr_module = cont_extr_module

    def __call__(self, trial):
        level = trial.suggest_float("level", self.min_level, self.max_level)
        self.cont_extr_module.run(level_value=level)
        return len(self.cont_extr_module.contours)

def optimize_cont_extr_hyperparams(dataset: HyperSpectralData, params: Params):
    """optimization only for 1 band for target class"""
    study = optuna.create_study(direction="maximize")
    study.optimize(ContExtrObjective(min_level=0.0, 
                                     max_level=10000.0,
                                     img=dataset.samples[0].band_img[0],
                                     cont_extr_module=OnnContourExtractionModule("ContourExtr")))
    
    for key, value in study.best_params:
        params.dict[key] = value

def optimize_hyperparams(onn_pipe: OnnHyperPipeline):
    for module in onn_pipe.model.modules.values():
        module.optimize_hyperparams(onn_pipe.dataset, onn_pipe.params)
