from dataset_loader import get_dataset
from datasets import HyperSpectralData
from onn_model import OnnModel
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
        self.result: np.array

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

    def create_samples(self):
        """some txt"""
        self.dataset.create_samples()

    def select_chanels(self, method = "expert"):
        """some txt"""
        self.dataset.select_chanels(method)

    def add_model(self, model):
        """some txt"""
        self.model = model

    def run(self, target_class: str, method: str = "expert"):
        """some txt"""
        self.specify_target_class(target_class)
        self.select_chanels(method)
        self.create_samples()

        self.result = self.model.run(self.dataset)
