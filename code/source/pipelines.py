from source.dataset_loader import get_dataset
from source.datasets import HyperSpectralData
from source.onn_model import OnnModel
from source.hyperparams_opt import optimize_hyperparams
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

    def select_chanels(self, method = "expert", num_of_bands = 4):
        """some txt"""
        self.dataset.select_chanels(method, num_of_bands)

    def add_model(self, model):
        """some txt"""
        self.model = model

    def run(self, 
            target_class: str,
            params_path: str = "",
            optimize_before_run:bool = False):
        """method to fully setup and run onn models"""
        self.specify_target_class(target_class)

        if len(params_path) != 0 and os.path.exists(params_path):
            file_path = params_path
        else:
            file_path = check_default_json_file()
        self.params = Params(file_path)

        self.select_chanels(self.params.band_sel_method, self.params.num_of_bands)
        self.create_samples(num_samples=self.params.num_samples,
                            sample_height=self.params.sample_height,
                            sample_width=self.params.sample_width,
                            threshold=self.params.threshold)
        
        if optimize_before_run:     
            optimize_hyperparams(self)
            self.params.update(file_path)

        self.model.run(self.dataset, 
                       po_num=self.params.po_num, 
                       params_sel_method=self.params.params_sel_method,
                       stimuls_num=self.params.stimuls_num,
                       stimuls_sel_method=self.params.stimuls_sel_method,
                       find_cont_method=self.params.find_cont_method, 
                       osc_params=[self.params.dict[f"stimul_{i}"] for i in range(self.params.stimuls_num)],
                       draw_contours=self.params.draw_contours,
                       level_value=self.params.level_value,
                       k_size=self.params.k_size,
                       max_number_of_iters=self.params.max_number_of_iters,
                       alpha=self.params.alpha,
                       beta=self.params.beta,
                       w1=self.params.w1,
                       w2_alpha=self.params.w2_alpha,
                       w2_beta=self.params.w2_beta,
                       w3_alpha=self.params.w3_alpha, 
                       w3_beta=self.params.w3_beta,
                       w4=self.params.w4,
                       threshold=self.params.threshold,
                       cont_area_threshold_percent=self.params.cont_area_threshold_percent)
    
    def eval(self, metric:str = "iou"):
        """some txt"""
        samples_result = evaluate_best_segmentation(self.model.segmented_samples,
                                                    self.dataset.samples,
                                                    self.dataset.target_class_id,
                                                    metric)
        self.result = samples_result
    
    def show_results(self, output_file):
        """some txt"""
        show_results(img=self.dataset.samples[0].original_img, 
                     segmented=self.result[0], 
                     gt=self.dataset.samples[0].labels,
                     rgb_bands = self.dataset.rgb,
                     output_file = output_file)
