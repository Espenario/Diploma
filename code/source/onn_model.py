from copy import deepcopy
from typing import Dict, Any
from skimage import measure
import numpy as np
from source.datasets import HyperSpectralData
from source.hyperperams_opt import optimize_cont_extr_hyperparams
from source.utils import *
from source.oscillators import CentralOscillator, PeripheralOscillator
from source.stimul import Stimul


class OnnModule():

    def __init__(self, module_name):
        """some txt"""
        self.module_name = module_name

    def run(self, img):
        """somt txt"""
        pass

    def optimize_hyperparams(self, dataset: HyperSpectralData):
        """some txt"""
        pass


class OnnContourExtractionModule(OnnModule):

    def __init__(self, module_name):
        super().__init__(module_name)
        self.contours: np.ndarray
        self.best_params: Dict[str, Any]
    
    def run(self, 
            img: np.ndarray, 
            find_cont_method:str = "library", 
            draw_contours:bool = True,
            level_value:str = 0.7,
            cont_area_threshold_percent:int = 1):
        """some txt
        Args:
            img: 2D np.array of shape HxW
        Return:
            3D np.array of representing contour lines for each spectral band
        """
        if find_cont_method == "library":
            self.extract_cont_library(img, level_value=level_value)

        if draw_contours:
            try:
                show_contours(img, self.contours)
            except KeyError:
                print("Extract contours module doesnt detect any contours( Try to change params")  # сделать такие сообщения как логи

        self.postprocess_contours(img_shape=img.shape, 
                                  cont_area_threshold_percent=cont_area_threshold_percent)

        return self.contours
    
    def extract_cont_library(self, img: np.ndarray, level_value:float=None):
        """some txt"""
        contours = measure.find_contours(img, level=level_value)
        self.contours = contours

    def postprocess_contours(self, 
                             img_shape: list,
                             cont_area_threshold_percent:int = 1):
        """some txt"""
        actual_contours = []
        cont_area_threshold = (img_shape[0] * img_shape[1]) / 100 * cont_area_threshold_percent
        for contour in self.contours:
            if find_contour_area(contour) > cont_area_threshold:
                actual_contours.append(contour)
        self.contours = actual_contours

    def optimize_hyperparams(self, dataset: HyperSpectralData, params: Params):
        optimize_cont_extr_hyperparams(dataset=dataset)
        self.level = params.level_value
        self.cont_area_threshold_percent = params.cont_area_threshold_percent



class OnnSelectiveAttentionModule2D(OnnModule):

    def __init__(self, module_name, att_method = "separate"):
        super().__init__(module_name)
        self.synchronization_states: list[list] = []
        self.central_oscillator: CentralOscillator
        self.periferal_oscillators: list[list[PeripheralOscillator]] = []
        self.stimuls: list[Stimul]
        self.att_method: str = att_method

    def run(self, 
            img:np.ndarray, 
            po_num:int = 2, 
            params_sel_method:str = "expert",
            stimuls_num:int = 2,
            stimuls_sel_method:str = "brightness",
            target_brightness:int = 0) -> np.ndarray:
        """some txt
        Args:
            img: 3D np.array of shape CxHxW
        Return:
            3D np.array of image shape with 0 representiong pixels without attention
        """
        selected_area = []
        for spectral_band_img in img:
            self.setup_oscillators(img=spectral_band_img, 
                                   po_num=po_num, 
                                   params_sel_method=params_sel_method,
                                   stimuls_num=stimuls_num,
                                   stimuls_sel_method=stimuls_sel_method,
                                   target_brightness=target_brightness)
            new_selection = self.perform_selection()
            if not(isinstance(new_selection, int)):
                if self.att_method == "separate":
                    selected_area.append(new_selection)
                if self.att_method == "intersect":
                    if len(selected_area) == 0:
                        selected_area = new_selection
                    else:
                        selected_area[selected_area != new_selection] = 0
                
        # if self.att_method == "separate":
        #     return np.asarray(selected_area)
        # if self.att_method == "intersect":
        if len(selected_area) == 0:
            return img
        return np.asarray(selected_area)
    
    def generate_oscillators_params(self, method:str = "expert", target_brightness:int = 0):
        if method == "expert":
            # Same params for all
            params = np.ones(len(self.stimuls)) * 10

            # bigger param for closest to target stimul
            stimul_values = np.array([x.stimul_values for x in self.stimuls])
            idx = (np.abs(stimul_values - target_brightness)).argmin()
            params[idx] = params[idx] * 15 + 23
            
        return params
    
    def setup_oscillators(self, 
                          img: np.array, 
                          po_num:int = 2, 
                          params_sel_method:str = "expert",
                          stimuls_num:int = 2,
                          stimuls_sel_method:str = "brightness",
                          target_brightness:int = 0):
        """select area of interest (where approximately target is located)"""
        self.periferal_oscillators = []
        self.stimuls = extract_stimuls(img=img,
                                       stimuls_num=stimuls_num,
                                       method=stimuls_sel_method)

        co_freq = np.mean(np.asarray(list(map(lambda x: x.stimul_values, self.stimuls)), dtype="object"))
        params = self.generate_oscillators_params(method=params_sel_method, 
                                                  target_brightness=target_brightness)

        self.central_oscillator = CentralOscillator(freq=co_freq,
                                                    phase=1,
                                                    params=params)
        
        for i, stimul in enumerate(self.stimuls):
            self.periferal_oscillators.append([])
            self.generate_periferal_oscillators(stimul=stimul, 
                                                stimul_id=i, 
                                                alpha=params[i],
                                                n=po_num)

        self.periferal_oscillators = np.asarray(self.periferal_oscillators)
        
    def generate_periferal_oscillators(self, stimul: Stimul, alpha:int, stimul_id:int, n:int = 2):
        """some_txt
        Args:
            n: number of oscillators per stimul
        """
        coef = 1e-4
        for i in range(min(n, len(stimul.stimul_values))):
            self.periferal_oscillators[stimul_id].append(PeripheralOscillator(phase=0,
                                                                              freq=stimul.stimul_values.mean() + coef * stimul.stimul_values[i],
                                                                              alpha=alpha
                                                                             ))
        
    def get_synchonization_state(self):
        """some txt"""
        for ensemple_po in self.periferal_oscillators:
            states = [oscillator.get_synchonization_state(self.central_oscillator)
                             for oscillator in ensemple_po]
            self.synchronization_states.append(states)

    def check_synchronization_state(self):
        """Check sync state by sum of every stimul PO ensemble states (each PO state could be 1 or 0)
        Returns: one of 3 possible syncronization states: ns (no sync), ps (part sync), gs (global sync)
        and id of sync state (relevant only for ps status, for other return -1)
        """
        self.synchronization_states = []
        self.get_synchonization_state()
        state_flag = 0
        state_id = -1
        for i, sync_state in enumerate(self.synchronization_states):
            if sum(sync_state) == len(sync_state):
                state_flag += 1
                state_id = i

        if state_flag == 0:
            return "ns", state_id
        if state_flag != len(self.synchronization_states):
            return "ps", state_id
        return "gs", state_id

    def perform_selection(self):
        """some txt"""    
        iter_num = 0
        while ((self.check_synchronization_state()[0] != "ps" and \
               self.check_synchronization_state()[0] != "gs") or iter_num < 1) and iter_num < 100:
            buf_co = deepcopy(self.central_oscillator)
            self.central_oscillator.step(self.periferal_oscillators)
            # print(iter_num)
            for ensemble in self.periferal_oscillators:
                for po in ensemble:
                    po.step(buf_co)
            # map(lambda x: x.step(buf_co), np.asarray(self.periferal_oscillators).flatten())
            # print(self.check_synchronization_state()[0], self.central_oscillator.phase, self.periferal_oscillators[0][0].phase, 
            #                                      self.periferal_oscillators[0][1].phase,
            #                                      self.periferal_oscillators[1][0].phase)
            iter_num += 1
        
        final_sync_state, id = self.check_synchronization_state()

        # print(final_sync_state)

        if final_sync_state == "ps":
            return self.stimuls[id].sub_img
        if final_sync_state == "gs":
            return self.stimuls[0].begin_img
        return 0

class OnnModel():

    def __init__(self, modules:dict = {}, model_name = "3_module_Onn") -> None:
        """some txt"""
        self.model_name = model_name
        self.modules:Dict[str, OnnModule] = modules
        self.segmented_samples: list[dict]

    def run(self, dataset):
        """some txt"""
        pass

    def add_module(self, module: OnnModule):
        self.modules[module.module_name] = module


class OnnModel2D(OnnModel):
    """some txt"""

    def __init__(self, modules:dict = {}, model_name = "3_module_Onn") -> None:
        """some txt"""
        super().__init__(modules, model_name)

    
    def run(self, 
            dataset: HyperSpectralData,
            po_num:int = 2, 
            params_sel_method:str = "expert",
            stimuls_num:int = 2,
            stimuls_sel_method:str = "brightness",
            find_cont_method:str = "library", 
            draw_contours:bool = True,
            level_value:str = None,
            cont_area_threshold_percent:int = 1):
        """some txt"""
        segmented_samples = []

        if "SelectiveAtt" not in self.modules.keys():
            raise KeyError("Module SelectiveAtt are not in model modules list, but is necessary")
              
        for sample in dataset.samples:
            area_of_interest_mask = self.modules["SelectiveAtt"].run(img=sample.band_img,
                                                                     po_num=po_num, 
                                                                     params_sel_method=params_sel_method, 
                                                                     stimuls_num=stimuls_num,
                                                                     stimuls_sel_method=stimuls_sel_method,
                                                                     target_brightness=sample.target_brightness)

            if self.modules["SelectiveAtt"].att_method == "separate":
                iterate_bands_over = area_of_interest_mask

            if self.modules["SelectiveAtt"].att_method == "intersect":
                iterate_bands_over = sample.band_img

            segmented_on_bands = {}
            for i, spectral_band_mask in enumerate(iterate_bands_over):
                
                area_of_interest = deepcopy(sample.band_img[i])
                if self.modules["SelectiveAtt"].att_method == "separate":
                    # verify that only current stimul represented in area_of_interest brightness values
                    area_of_interest[area_of_interest != spectral_band_mask] = -1

                if self.modules["SelectiveAtt"].att_method == "intersect":
                    # verify that only current stimul represented in area_of_interest brightness values
                    area_of_interest[area_of_interest != area_of_interest_mask] = -1

                # crop the area to minimuze furthewr calculations
                area_of_interest = area_of_interest[~np.all(area_of_interest == -1, axis=1)]
                area_of_interest = area_of_interest[:,~(area_of_interest==-1).all(0)]

                try:
                    contours = self.modules["ContourExtr"].run(img=area_of_interest,
                                                               find_cont_method=find_cont_method, 
                                                               draw_contours=draw_contours,
                                                               level_value=level_value,
                                                               cont_area_threshold_percent=cont_area_threshold_percent)
                except KeyError:
                    segmented_on_bands[dataset.selected_bands[i]] = area_of_interest
                    continue

                try:
                    segmented_img = self.modules["Segmentation"].run(contours, area_of_interest)
                except KeyError:
                    segmented_on_bands[dataset.selected_bands[i]] = area_of_interest
                    continue

                segmented_on_bands[dataset.selected_bands[i]] = segmented_img

            segmented_samples.append(segmented_on_bands)

        self.segmented_samples = segmented_samples