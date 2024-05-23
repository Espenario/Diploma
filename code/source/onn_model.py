from copy import deepcopy
from typing import Dict, Any
from skimage import measure
import numpy as np
from source.datasets import HyperSpectralData
from source.hyperperams_opt import optimize_cont_extr_hyperparams, optimize_sel_att_hyperparams
from source.utils import *
from source.GaborGradScale import GaborGradScale
from source.oscillators import *
from source.stimul import Stimul


class OnnModule():

    def __init__(self, module_name):
        """some txt"""
        self.module_name = module_name

    def run(self, img):
        """somt txt"""
        pass

    def optimize_hyperparams(self, dataset: HyperSpectralData, params: Params):
        """some txt"""
        pass


class OnnSegmentationModule(OnnModule):

    def __init__(self, module_name):
        super().__init__(module_name)
        self.layer1: np.ndarray[np.ndarray[PeripheralOscillatorSegmentationL1]]
        self.layer2: np.ndarray[np.ndarray[PeripheralOscillatorSegmentationL2]]
        self.central_oscillator: CentralOscillatorSegmentation
        self.s_area_oscillators: np.ndarray[PeripheralOscillatorSegmentationL1]
        self.img: np.ndarray
        self.gt: np.ndarray
        self.contours: np.ndarray
        self.number_of_iters: int = 0
        self.begin_part_size: int = 25
        self.begin_part_coord: list
        self.w1: float
        self.w2: callable
        self.w3: callable
        self.w4: float
        self.w5: callable
        self.g: callable
        self.alpha: float
        self.beta: float

    def run(self,
            img: np.ndarray,
            gt: np.ndarray,
            contours: np.ndarray):
        """some txt
        Args:
            img: 2D np.ndarray HxW, sample image of 1 band, values are braightnesses in this band
            contours: 2D np.ndarray HxW, contours mask, pixel_value is 1 if there ara contour point, 0 otherwise
        """
        self.img = img
        self.gt = gt
        self.contours = contours
        self.number_of_iters = 0
        self.setup_params()
        self.setup_oscillators()
        while self.check_stop_condition():

            self.central_oscillator.step(s_area_osc=self.s_area_oscillators, 
                                         w1=self.w1, 
                                         alpha=self.alpha,
                                         g=self.g)
            
            for i, row in enumerate(self.layer1):
                for j, _ in enumerate(row):
                    neibours = self.get_all_neibours([i, j])
                    self.layer1[i, j].step(central_oscillator = self.central_oscillator, 
                                           w2 = self.w2, 
                                           w3 = self.w3, 
                                           point = [i, j], 
                                           s_size = self.begin_part_size,
                                           s_start = self.begin_part_coord,
                                           neibours = neibours[0],
                                           t = self.number_of_iters)
                    
                    senders_to_l2 = self.get_senders_to_l2([i, j])
                    self.layer2[i, j].step(w4 = self.w4,
                                           neibours = neibours[1],
                                           senders_to_l2 = senders_to_l2, 
                                           w5 = self.w5,
                                           point = [i, j],
                                           beta = self.beta,
                                           t = self.number_of_iters)

            self.number_of_iters += 1
            self.central_oscillator.update()
            for i, row in enumerate(self.layer1):
                for j, _ in enumerate(row):
                    self.layer1[i, j].update()
                    self.layer2[i, j].update()

    def get_all_neibours(self, point):
        """for l1 is 8 neighbour and for l2
        returns:
            neighbours: tuple of PO objects, points(i, j) for l1 and l2 
        """
        row, col = point
        neighbors_l1 = []
        neighbors_l2 = []
        for i in range(max(0, row-1), min(self.layer1.shape[0], row+2)):
            for j in range(max(0, col-1), min(self.layer1.shape[1], col+2)):
                if (i, j) != (row, col):
                    neighbors_l1.append((self.layer1[i, j], (i, j)))
                    neighbors_l2.append((self.layer2[i, j], (i, j)))

        return (neighbors_l1, neighbors_l2)

    def get_senders_to_l2(self, point):
        """mat 7x7 from l1"""
        row, col = point
        senders_to_l2 = []
        for i in range(max(0, row-3), min(self.layer1.shape[0], row+4)):
            for j in range(max(0, col-3), min(self.layer1.shape[1], col+4)):
                senders_to_l2.append((self.layer1[i, j], (i, j)))

        return senders_to_l2
            
    def check_stop_condition(self):
        """some txt"""
        if self.number_of_iters <= 1000:
            return True
        return False
    
    def setup_params(self):
        """some txt"""
        self.alpha = 2
        self.beta = 3
        self.w1 = 10
        self.w4 = 5
        self.w2 = linear_descending_to_0
        self.w3 = square_ascending
        self.w5 = exp_dec_with_distance
        self.g = g

    def setup_oscillators(self):
        """some txt"""
        # fill both layers
        self.layer1 = np.vectorize(lambda x: PeripheralOscillatorSegmentationL1(freq=np.random.uniform(4, 5),
                                                                              phase=np.random.uniform(0, 0.2),
                                                                              level = 1))(self.img)
        self.layer2 = np.vectorize(lambda x: PeripheralOscillatorSegmentationL2(freq=np.random.uniform(4, 5),
                                                                              phase=np.random.uniform(0, 0.2),
                                                                              level = 2))(self.img)
        self.central_oscillator = CentralOscillatorSegmentation(freq=6, phase=0)
 
        #setup inactive border oscillators
        self.layer1 = np.where(self.contours == 1, self.layer1.disable(), self.layer1)

        #setup begin square, from which segmentation starts
        begin_part_coord, begin_part_size = find_largest_square(self.gt)
        begin_part_size -= 1
        begin_part_coord[0] += 1
        begin_part_coord[1] += 1
        self.begin_part_coord = begin_part_coord
        self.begin_part_size = begin_part_size**2
        
        #increase freq of begin square oscillators and get s_oscillators
        increase_value = 4
        s_area_oscillators = []
        for i, row in enumerate(self.layer1):
            for j, _ in enumerate(row):
                if i > begin_part_coord[0] and i < begin_part_coord[0] + begin_part_size and \
                   j > begin_part_coord[1] and j < begin_part_coord[1] + begin_part_size:
                    self.layer1[i, j].freq += increase_value
                    s_area_oscillators.append(self.layer1[i, j])

        self.s_area_oscillators = np.array(s_area_oscillators)


class OnnContourExtractionModule(OnnModule):

    def __init__(self, module_name):
        super().__init__(module_name)
        self.contours: np.ndarray
        self.best_params: Dict[str, Any]
    
    def run(self, 
            img: np.ndarray, 
            find_cont_method:str = "library", 
            draw_contours:bool = False,
            level_value:str = 0.7,
            target_class_brightness: float = 100,
            cont_area_threshold_percent:int = 1):
        """some txt
        Args:
            img: 2D np.array of shape HxW
        Return:
            3D np.array of representing contour lines for each spectral band
        """
        if find_cont_method == "library":
            self.extract_cont_library(img, level_value=level_value)

        if find_cont_method == "gabor_grad_scale":
            self.extract_cont_gabor_grad_scale(img, target_class_brightness)

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

    def extract_cont_gabor_grad_scale(self, img: np.ndarray, target_class_brightness: float):
        """some txt"""
        contours = GaborGradScale(img, target_class_brightness).extract_contours()
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
        optimize_cont_extr_hyperparams(cont_extr_module=self, dataset=dataset, params=params)
        self.level = params.level_value
        self.cont_area_threshold_percent = params.cont_area_threshold_percent


class OnnSelectiveAttentionModule2D(OnnModule):

    def __init__(self, module_name, att_method = "separate"):
        super().__init__(module_name)
        self.synchronization_states: list[list] = []
        self.central_oscillator: CentralOscillatorSelAtt
        self.periferal_oscillators: list[list[PeripheralOscillatorSelAtt]] = []
        self.stimuls: list[Stimul]
        self.att_method: str = att_method

    def run(self, 
            img:np.ndarray, 
            po_num:int = 2, 
            params_sel_method:str = "simple_random",
            stimuls_num:int = 2,
            stimuls_sel_method:str = "brightness",
            osc_params:int = [],
            target_brightness:list = []) -> np.ndarray:
        """some txt
        Args:
            img: 3D np.array of shape CxHxW
        Return:
            3D np.array of image shape with 0 representiong pixels without attention
        """
        selected_area = []
        for i, spectral_band_img in enumerate(img):
            self.setup_oscillators(img=spectral_band_img, 
                                   po_num=po_num, 
                                   params_sel_method=params_sel_method,
                                   stimuls_num=stimuls_num,
                                   stimuls_sel_method=stimuls_sel_method,
                                   osc_params=osc_params,
                                   target_brightness=target_brightness[i])
            new_selection = self.perform_selection()
            if self.att_method == "separate":
                selected_area.append(new_selection)
            if self.att_method == "intersect":
                if len(selected_area) == 0:
                    selected_area = new_selection
                else:
                    selected_area[selected_area != new_selection] = 0

        selected_area = np.asarray(selected_area)

        if len(selected_area) == 0:
            return img
        return selected_area
    
    def generate_oscillators_params(self, method:str = "simple_random", exp_params:list = [], target_brightness:int = 0):
        if method == "expert" and len(exp_params) == len(self.stimuls):
            params = exp_params

        if method == "simple_random":
            # Same params for all
            params = np.ones(len(self.stimuls)) * 10

            # bigger param for closest to target stimul
            stimul_values = np.array([x.stimul_values.mean() for x in self.stimuls])
            idx = (np.abs(stimul_values - target_brightness)).argmin()
            params[idx] = params[idx] * 15 + 23

        return params
    
    def setup_oscillators(self, 
                          img: np.array, 
                          po_num:int = 2, 
                          params_sel_method:str = "simple_random",
                          stimuls_num:int = 2,
                          stimuls_sel_method:str = "brightness",
                          osc_params:list = [],
                          target_brightness:int = 0):
        """select area of interest (where approximately target is located)"""
        self.periferal_oscillators = []
        self.stimuls = extract_stimuls(img=img,
                                       stimuls_num=stimuls_num,
                                       method=stimuls_sel_method)

        co_freq = np.mean(np.asarray(list(map(lambda x: x.stimul_values, self.stimuls)), dtype="object"))
        params = self.generate_oscillators_params(method=params_sel_method, 
                                                  target_brightness=target_brightness,
                                                  exp_params=osc_params)

        self.central_oscillator = CentralOscillatorSelAtt(freq=co_freq,
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
            self.periferal_oscillators[stimul_id].append(PeripheralOscillatorSelAtt(phase=0,
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
        
        return self.stimuls[0].begin_img
    
    def optimize_hyperparams(self, dataset: HyperSpectralData, params: Params):
        optimize_sel_att_hyperparams(self, dataset=dataset, params=params)

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
            osc_params: list = [],
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
                                                                     osc_params=osc_params,
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

                # crop the area to minimuze further calculations
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