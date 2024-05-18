import numpy as np


class Stimul():

    def __init__(self, img, sub_img, stimul_values):
        """some txt"""
        self.sub_img:np.ndarray = sub_img
        self.stimul_values:np.ndarray = stimul_values
        self.begin_img:np.ndarray = img

    