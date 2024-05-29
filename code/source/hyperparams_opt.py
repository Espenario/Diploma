import optuna
import numpy as np
from source.utils import *
from source.datasets import HyperSpectralData

class ContExtrObjective:

    def __init__(self, 
                 min_level:float, 
                 max_level:float,
                 min_k_size:int,
                 max_k_size:int,
                 img:np.ndarray,
                 cont_extr_module,
                 cont_area_threshold_percent_min:float = 1,
                 cont_area_threshold_percent_max:float = 10):

        self.min_level = min_level
        self.max_level = max_level
        self.min_k_size = min_k_size
        self.max_k_size = max_k_size
        self.img = img
        self.cont_area_threshold_percent_min = cont_area_threshold_percent_min
        self.cont_area_threshold_percent_max = cont_area_threshold_percent_max
        self.cont_extr_module = cont_extr_module

    def __call__(self, trial: optuna.trial):
        find_cont_method = trial.suggest_categorical("find_cont_method", ["simple_sobel", "gabor_grad_scale"])
        self.cont_extr_module.run(img=self.img, find_cont_method=find_cont_method)
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

        # self.best_resres[0].sample[res[0].best_band_id].shape)
        return res[0].best_score
    
class SegmentationObjective:

    def __init__(self,
                 max_number_of_iters,
                 min_number_of_iters,
                 max_w1,
                 min_w1,
                 max_alpha, 
                 min_alpha,
                 max_beta, 
                 min_beta,
                 max_w4,
                 min_w4,
                 sample,
                 contours,
                 segm_module):
        self.max_number_of_iters = max_number_of_iters
        self.min_number_of_iters = min_number_of_iters
        self.max_w1 = max_w1
        self.min_w1 = min_w1
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.max_w4 = max_w4
        self.min_w4 = min_w4
        self.sample = sample
        self.contours = contours
        self.segm_module = segm_module

    def __call__(self, trial):
        number_of_iters = trial.suggest_int("max_number_of_iters", 
                                            self.min_number_of_iters, 
                                            self.max_number_of_iters)
        w1 = trial.suggest_int("w1", self.min_w1, self.max_w1)
        alpha = trial.suggest_int("alpha", self.min_alpha, self.max_alpha)
        beta = trial.suggest_int("beta", self.min_beta, self.max_beta)
        w4 = trial.suggest_int("w4", self.min_w4, self.max_w4)
        threshold = trial.suggest_int("threshold_segm", 10, 100)
        increase_value = trial.suggest_float("increase_value", 0.1, 20)
        w2_alpha = trial.suggest_float("w2_alpha", -1, -0.1)
        w2_beta = trial.suggest_int("w2_beta", 1, 10)
        w3_alpha = trial.suggest_float("w3_alpha", 0.1, 1)
        w3_beta = trial.suggest_int("w3_beta", 5, 35)
        band_id = trial.suggest_int("band_id", 0, 23)
        segm_res = self.segm_module.run(img = self.sample.band_img[band_id],
                                        gt = self.sample.labels,
                                        contours = self.contours,
                                        max_number_of_iters=7,
                                        increase_value=increase_value,
                                        alpha=alpha,
                                        beta=beta,
                                        w1=w1,
                                        w2_alpha=w2_alpha,
                                        w2_beta=w2_beta,
                                        w3_alpha=w3_alpha,
                                        w3_beta=w3_beta,
                                        w4=w4,
                                        threshold=30)
    
        segmented_on_bands = {}
        segmented_on_bands[0] = segm_res
        # cv2.imwrite("best_sample.png", segm_res)
        # cv2.imwrite("orig.png", self.sample.band_img[0])
        target_class_id = self.sample.labels[np.nonzero(self.sample.labels)][0]
        res = evaluate_best_segmentation([segmented_on_bands], [self.sample], target_class_id)

        return res[0].best_score

def optimize_segmentation_hyperparams(segm_module, cont_extr_module, dataset: HyperSpectralData, params: Params):
    """optimization of params of segmentation module"""
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    test_img = dataset.samples[0].band_img[0]
    contours = cont_extr_module.run(img=test_img)
    study.optimize(SegmentationObjective(max_number_of_iters = 10,
                                         min_number_of_iters = 1,
                                         max_w1 = 20,
                                         min_w1 = 1,
                                         max_alpha = 20,
                                         min_alpha = 1,
                                         max_beta = 20,
                                         min_beta = 1,
                                         max_w4 = 20,
                                         min_w4 = 1,
                                         sample=dataset.samples[0],
                                         contours=contours,
                                         segm_module=segm_module), n_trials=1000, n_jobs=1)

    for key, value in study.best_params.items():
        params.dict[key] = value


def optimize_cont_extr_hyperparams(cont_extr_module, dataset: HyperSpectralData, params: Params):
    """optimization only for 1 band for target class"""
    study = optuna.create_study(direction="maximize")
    study.optimize(ContExtrObjective(min_level=0.0, 
                                     max_level=10000.0,
                                     min_k_size=1,
                                     max_k_size=7,
                                     img=dataset.samples[0].band_img[0],
                                     cont_extr_module=cont_extr_module),
                                     n_trials=10)
    
    study.best_value
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

# def optimize_hyperparams_model(onn_pipe):
#     onn_pipe.model.modules["SelectiveAtt"].optimize_hyperparams(onn_pipe.dataset, onn_pipe.params)
#     focus_img = 

#     dir = HYPERPARAMS_DIR
#     file_name = onn_pipe.model.model_name

#     files_in_dir = os.listdir(dir)

#     max_idx = 0
#     for f in files_in_dir:
#         if f.startswith(file_name):
#             last_part = f.split('_')[-1]
#             idx = int(last_part[:last_part.find(".")])
#             if idx > max_idx:
#                 max_idx = idx
    
#     file_path = os.path.join(dir, f"{file_name}_{max_idx + 1}.json")
    
#     onn_pipe.params.save(file_path)

def optimize_hyperparams(onn_pipe):
    for module in onn_pipe.model.modules.values():
        if module.module_name == "Segmentation":
            module.optimize_hyperparams(onn_pipe.model.modules["ContourExtr"],
                                        onn_pipe.dataset, 
                                        onn_pipe.params)
        else:
            module.optimize_hyperparams(onn_pipe.dataset, 
                                        onn_pipe.params)

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
