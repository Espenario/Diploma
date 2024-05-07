import numpy as np


class OnnModel():
    """some txt"""

    def __init__(self, model_name = "3_module_Onn") -> None:
        """some txt"""
        self.model_name = model_name
        pass
    
    def run(self, dataset: dict):
        """some txt"""
        print(dataset)
        return np.asarray([self.model_name, dataset.keys()])