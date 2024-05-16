import numpy as np
from source.datasets import HyperSpectralData
from source.utils import extract_stimuls
from source.oscillators import CentralOscillator, PeripheralOscillator
from copy import deepcopy


class OnnModule():

    def __init__(self, module_name):
        """some txt"""
        self.module_name = module_name

    def run(self, img):
        """somt txt"""
        pass


class OnnSelectiveAttentionModule2D(OnnModule):

    def __init__(self, module_name):
        super().__init__(module_name)
        self.synchronization_states: list[list] = []
        self.central_oscillator: CentralOscillator
        self.periferal_oscillators: list[list[PeripheralOscillator]] = []

    def run(self, img: np.array) -> np.array:
        """some txt"""
        selected_area = []
        for spectral_band_img in img:
            self.setup_oscillators(spectral_band_img)
            selected_area.append(self.perform_selection())
        return selected_area
    
    def setup_oscillators(self, img: np.array):
        """select area of interest (where approximately target is located)"""
        stimuls, co_freq = extract_stimuls(img)
        self.central_oscillator = CentralOscillator(freq=co_freq,
                                                    phase=0,
                                                    alpha=np.random.randn(),
                                                    beta=np.random.randn())
        for i, stimul in enumerate(stimuls):
            self.generate_periferal_oscillators(i, stimul)
        
    def generate_periferal_oscillators(self, stimul: np.ndarray, stimul_id:int, n:int = 20):
        """some_txt
        Args:
            n: number of oscillators per stimul
        """
        for _ in range(n):
            self.periferal_oscillators[stimul_id].append(PeripheralOscillator(phase=0,
                                                                              freq=stimul.mean(),
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
        self.get_synchonization_state()
        state_flag = 0
        state_id = -1
        for i, sync_state in enumerate(self.synchronization_states):
            if sum(sync_state) == len(sync_state):
                state_flag += 1
                state_id = i
        
        if state_flag == 0:
            return "ns", state_id
        if state_flag == 1:
            return "ps", state_id
        return "gs", state_id

    def perform_selection(self):
        """some txt"""    
        iter_num = 0
        while self.check_synchronization_state()[0] == "ns" and iter_num < 10000:
            buf_co = deepcopy(self.central_oscillator)
            self.central_oscillator.step()
            map(lambda x: x.step(buf_co), np.asarray(self.periferal_oscillators).flatten())
            

class OnnModel():

    def __init__(self, modules:dict, model_name = "3_module_Onn") -> None:
        """some txt"""
        self.model_name = model_name
        self.modules = modules
        self.segmented_samples: list[dict]

    def run(self, dataset):
        """some txt"""
        pass


class OnnModel2D(OnnModel):
    """some txt"""

    def __init__(self, modules:dict, model_name = "3_module_Onn") -> None:
        """some txt"""
        super().__init__(modules, model_name)

    
    def run(self, dataset: HyperSpectralData):
        """some txt"""
        segmented_samples = []
        for sample in dataset.samples:
            segmented_on_bands = {}
            area_of_interest = self.modules["SelectiveAtt"].run(sample)
            for spectral_band in area_of_interest:
                contours = self.modules["ContourExtr"].run(area_of_interest)
                segmented_img = self.modules["Segmentation"].run(contours, area_of_interest)
                segmented_on_bands[spectral_band] = segmented_img

            segmented_samples.append(segmented_on_bands)

        self.segmented_samples = segmented_samples