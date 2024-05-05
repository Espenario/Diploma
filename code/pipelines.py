from dataset_loader import get_dataset
from datasets import HyperSpectralData


class HyperPipeline(): 
    """Base class for models pipeline"""

    def __init__(self, dataset_name = 'PaviaU', load_folder = "./datasets"):
        self.dataset_name = dataset_name
        self.load_folder = load_folder
    
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

    def __init__(self, dataset_name='PaviaU', load_folder="./datasets"):
        super().__init__(dataset_name, load_folder)
        self.dataset: HyperSpectralData

    def load_and_prepare_data(self):
        """some txt"""
        data, gt, labels, ignored_labels, rgb_bands, palette, num_of_bands = get_dataset(self.dataset_name, self.load_folder)
        params = {'labels': labels, 
                  'ignored_labels': ignored_labels,
                  'rbg_bands': rgb_bands,
                  'palette': palette,
                  'num_of_bands': num_of_bands}
        self.dataset = HyperSpectralData(data, gt, **params)

    def get_data_info(self):
        """some txt"""
        self.dataset.get_data_info()
