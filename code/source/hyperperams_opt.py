import optuna
import numpy as np
from source.utils import *
from source.datasets import HyperSpectralData

class ContExtrObjective:

    def __init__(self, 
                 min_level:float, 
                 max_level:float, 
                 img:np.ndarray,
                 cont_extr_module,
                 cont_area_threshold_percent_min:float = 1,
                 cont_area_threshold_percent_max:float = 10):

        self.min_level = min_level
        self.max_level = max_level
        self.img = img
        self.cont_area_threshold_percent_min = cont_area_threshold_percent_min
        self.cont_area_threshold_percent_max = cont_area_threshold_percent_max
        self.cont_extr_module = cont_extr_module

    def __call__(self, trial: optuna.trial):
        level = trial.suggest_float("level_value", self.min_level, self.max_level)
        self.cont_extr_module.run(img=self.img, level_value=level)
        return len(self.cont_extr_module.contours)
    
class SelAttObjective:

    def __init__(self,
                 min_po_num:int,
                 max_po_num:int,
                 sample:np.ndarray,
                 sel_att_module,
                 selected_bands: list,
                 target_class_id: int,
                 min_stimuls:int,
                 max_stimuls:int):
        self.min_po_num = min_po_num
        self.max_po_num = max_po_num
        self.sample = sample
        self.target_class_id = target_class_id
        self.selected_bands = selected_bands
        self.sel_att_module = sel_att_module
        self.min_stimuls = min_stimuls
        self.max_stimuls = max_stimuls

    def __call__(self, trial: optuna.trial):
        po_num = trial.suggest_int("po_num", self.min_po_num, self.max_po_num)
        stimuls_num = trial.suggest_int("stimuls_num", self.min_stimuls, self.max_stimuls)
        params_sel_method = trial.suggest_categorical("params_sel_method", ["simple_random", "expert"])
        osc_params = [trial.suggest_int(f"stimul_{i}", 1, 10000) for i in range(stimuls_num)]
        area_of_interest_mask = self.sel_att_module.run(img=self.sample.band_img, 
                                                        po_num=po_num, 
                                                        params_sel_method=params_sel_method,
                                                        osc_params=osc_params,
                                                        stimuls_num=stimuls_num,
                                                        target_brightness=self.sample.target_brightness)
        
        segmented_on_bands = {}
        for i, spectral_band_mask in enumerate(area_of_interest_mask):  
            area_of_interest = deepcopy(self.sample.band_img[i])
            area_of_interest[area_of_interest != spectral_band_mask] = -1

            area_of_interest = area_of_interest[~np.all(area_of_interest == -1, axis=1)]
            area_of_interest = area_of_interest[:,~(area_of_interest==-1).all(0)]

            segmented_on_bands[self.selected_bands[i]] = area_of_interest

        res = evaluate_best_segmentation([segmented_on_bands], [self.sample], self.target_class_id)

        print(res[0].sample[res[0].best_band_id].shape)
        return res[0].best_score

def optimize_cont_extr_hyperparams(cont_extr_module, dataset: HyperSpectralData, params: Params):
    """optimization only for 1 band for target class"""
    study = optuna.create_study(direction="maximize")
    study.optimize(ContExtrObjective(min_level=0.0, 
                                     max_level=10000.0,
                                     img=dataset.samples[0].band_img[0],
                                     cont_extr_module=cont_extr_module),
                                     n_trials=10)
    
    for key, value in study.best_params.items():
        params.dict[key] = value

def optimize_sel_att_hyperparams(sel_att_module, dataset: HyperSpectralData, params: Params):
    """optimization of params of sel_att_module"""
    study = optuna.create_study(direction="maximize")
    study.optimize(SelAttObjective(min_po_num=1,
                                   max_po_num=50,
                                   sample=dataset.samples[0],
                                   sel_att_module=sel_att_module,
                                   selected_bands=dataset.selected_bands,
                                   target_class_id=dataset.target_class_id,
                                   min_stimuls=2,
                                   max_stimuls=17), n_trials=10)
    
    for key, value in study.best_params.items():
        params.dict[key] = value

def optimize_hyperparams(onn_pipe):
    for module in onn_pipe.model.modules.values():
        module.optimize_hyperparams(onn_pipe.dataset, onn_pipe.params)

    dir = HYPERPARAMS_DIR
    file_name = onn_pipe.model.model_name

    files_in_dir = os.listdir(dir)

    max_idx = 0
    for f in files_in_dir:
        if f.startswith(file_name):
            last_part = f.split('_')[-1]
            idx = int(last_part[:last_part.find(".")])
            if idx > max_idx:
                max_idx = idx
    
    file_path = os.path.join(dir, f"{file_name}_{max_idx + 1}.json")
    
    onn_pipe.params.save(file_path)
