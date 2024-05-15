import numpy as np
from source.datasets import HyperSpectralData


class OnnModel():
    """some txt"""

    def __init__(self, model_name = "3_module_Onn") -> None:
        """some txt"""
        self.model_name = model_name
    
    def run(self, dataset: HyperSpectralData):
        """some txt"""
        # print(dataset)
        return self.model_name, dataset.samples, dataset.samples_labels