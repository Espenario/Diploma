from source.dataset_loader import get_dataset
from source.datasets import HyperSpectralData
from source.onn_model import OnnModel
from source.hyperperams_opt import optimize_hyperparams
from source.utils import *
import numpy as np

class HyperPipeline():
    """Base class for models pipeline"""

    def __init__(self):
        """some txt"""
        pass
   
    def load_and_prepare_data(self):
        """some txt"""
        pass

    def get_data_info(self):
        """sone txt"""
        pass

    def train(self):
        """some txt"""
        pass

    def predict(self):
        """some txt"""
        pass

class OnnHyperPipeline(HyperPipeline):
    """Pipeline class for implementing ONN workflow"""

    def __init__(self):
        super().__init__()
        self.dataset: HyperSpectralData
        self.model: OnnModel
        self.result: list[Result]
        self.params: Params

    def add_dataset(self, dataset_name='PaviaU', load_folder="./datasets"):
        """some txt"""
        data, gt, labels, ignored_labels, \
        rgb_bands, palette, num_of_bands = get_dataset(dataset_name, load_folder)
        params = {'labels': labels, 
                  'ignored_labels': ignored_labels,
                  'rbg_bands': rgb_bands,
                  'palette': palette,
                  'num_of_bands': num_of_bands}
        self.dataset = HyperSpectralData(data, gt, **params)

    def get_data_info(self):
        """some txt"""
        self.dataset.get_data_info()
        self.dataset.view_data()
    
    def specify_target_class(self, target_class: str):
        """some txt"""
        self.dataset.specify_target_class(target_class)

    def create_samples(self, 
                       num_samples:int = 3,
                       sample_height:int = 100,
                       sample_width:int = 100,
                       threshold:int = 100):
        """some txt"""
        self.dataset.create_samples(num_samples=num_samples, 
                                    sample_height=sample_height,
                                    sample_width=sample_width,
                                    threshold=threshold)

    def select_chanels(self, method = "expert", n = 4):
        """some txt"""
        self.dataset.select_chanels(method, n)

    def add_model(self, model):
        """some txt"""
        self.model = model

    def run(self, 
            target_class: str, 
            band_sel_method: str = "expert", 
            num_samples:int = 3, 
            sample_height:int = 100,
            sample_width:int = 100,
            threshold:int = 100,
            po_num:int = 2, 
            params_sel_method:str = "expert",
            stimuls_num:int = 2,
            stimuls_sel_method:str = "brightness",
            find_cont_method:str = "library", 
            draw_contours:bool = True,
            level_value:str = None,
            cont_area_threshold_percent:int = 1,
            optimize_before_run:bool = False):
        """some txt
        Args:
            threshold: how much pixels of target class should have been in sample to accept it

        """
        self.specify_target_class(target_class)
        self.select_chanels(band_sel_method)
        self.create_samples(num_samples=num_samples, 
                            sample_height=sample_height,
                            sample_width=sample_width,
                            threshold=threshold)
        
        if optimize_before_run:
            file_path = check_default_json_file()
            self.params = Params(file_path)
            optimize_hyperparams(self)

        self.model.run(self.dataset, 
                       po_num=po_num, 
                       params_sel_method=params_sel_method,
                       stimuls_num=stimuls_num,
                       stimuls_sel_method=stimuls_sel_method,
                       find_cont_method=find_cont_method, 
                       draw_contours=draw_contours,
                       level_value=level_value,
                       cont_area_threshold_percent=cont_area_threshold_percent)
    
    def eval(self, metric:str = "iou"):
        """some txt"""
        samples_result = evaluate_best_segmentation(self.model.segmented_samples,
                                                    self.dataset.samples,
                                                    self.dataset.target_class_id,
                                                    metric)
        self.result = samples_result
    
    def show_results(self):
        """some txt"""
        show_results(img=self.dataset.samples[0].original_img, 
                     segmented=self.result[0], 
                     gt=self.dataset.samples[0].labels,
                     rgb_bands = self.dataset.rgb,
                     target_class = self.dataset.target_class)
