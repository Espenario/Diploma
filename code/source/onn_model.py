from copy import deepcopy
from typing import Dict
import numpy as np
from source.datasets import HyperSpectralData
from source.utils import extract_stimuls
from source.oscillators import CentralOscillator, PeripheralOscillator
from source.stimul import Stimul


class OnnModule():

    def __init__(self, module_name):
        """some txt"""
        self.module_name = module_name

    def run(self, img):
        """somt txt"""
        pass


class OnnSelectiveAttentionModule2D(OnnModule):

    def __init__(self, module_name, att_method = "separate"):
        super().__init__(module_name)
        self.synchronization_states: list[list] = []
        self.central_oscillator: CentralOscillator
        self.periferal_oscillators: list[list[PeripheralOscillator]] = []
        self.stimuls: list[Stimul]
        self.att_method: str = att_method

    def run(self, img: np.ndarray) -> np.ndarray:
        """some txt
        Args:
            img: 3D np.array of shape CxHxW
        Return:
            2D np.array of image shape with 1 representiong pixels in attention? 0 without attention
        """
        selected_area = []
        for spectral_band_img in img:
            self.setup_oscillators(spectral_band_img)
            new_selection = self.perform_selection()
            if ~isinstance(new_selection, int):
                if self.att_method == "separate":
                    selected_area.append(new_selection)
                if self.att_method == "intersect":
                    if len(selected_area) == 0:
                        selected_area = new_selection
                    else:
                        selected_area[selected_area == new_selection] = 1
                        selected_area[selected_area != new_selection] = 0
                
        # if self.att_method == "separate":
        #     return np.asarray(selected_area)
        # if self.att_method == "intersect":
        return selected_area
    
    def setup_oscillators(self, img: np.array):
        """select area of interest (where approximately target is located)"""
        self.periferal_oscillators = []
        self.stimuls = extract_stimuls(img)
        co_freq = np.mean(np.asarray(list(map(lambda x: x.stimul_values, self.stimuls)), dtype="object"))
        self.central_oscillator = CentralOscillator(freq=co_freq,
                                                    phase=0,
                                                    params=np.random.randn(len(self.stimuls)))
        
        for i, stimul in enumerate(self.stimuls):
            self.periferal_oscillators.append([])
            self.generate_periferal_oscillators(stimul, i)

        self.periferal_oscillators = np.asarray(self.periferal_oscillators)
        
    def generate_periferal_oscillators(self, stimul: Stimul, stimul_id:int, n:int = 20):
        """some_txt
        Args:
            n: number of oscillators per stimul
        """
        for _ in range(n):
            self.periferal_oscillators[stimul_id].append(PeripheralOscillator(phase=0,
                                                                              freq=stimul.stimul_values.mean(),
                                                                              alpha=np.random.randn()
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
        while self.check_synchronization_state()[0] == "ns" and iter_num < 10000:
            buf_co = deepcopy(self.central_oscillator)
            self.central_oscillator.step()
            map(lambda x: x.step(buf_co), np.asarray(self.periferal_oscillators).flatten())
        
        final_sync_state, id = self.check_synchronization_state()

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

    
    def run(self, dataset: HyperSpectralData):
        """some txt"""
        segmented_samples = []

        if "SelectiveAtt" not in self.modules.keys():
            raise KeyError("Module SelectiveAtt are not in model modules list, but is necessary")
              
        for sample in dataset.samples:
            area_of_interest_mask = self.modules["SelectiveAtt"].run(sample)

            if self.modules["SelectiveAtt"].att_method == "separate":
                iterate_bands_over = area_of_interest_mask

            if self.modules["SelectiveAtt"].att_method == "intersect":
                iterate_bands_over = sample

            segmented_on_bands = {}
            for i, spectral_band_mask in enumerate(iterate_bands_over):
                
                area_of_interest = sample[i]
                if self.modules["SelectiveAtt"].att_method == "separate":
                    area_of_interest[area_of_interest != spectral_band_mask] = -1

                if self.modules["SelectiveAtt"].att_method == "intersect":
                    area_of_interest[area_of_interest != area_of_interest_mask] = -1

                try:
                    contours = self.modules["ContourExtr"].run(area_of_interest)
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