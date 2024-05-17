import numpy as np


class Oscillator():

    def __init__(self, freq, phase):
        """some txt"""
        self.freq:float = freq
        self.phase:float = phase
    
    def step(self, oscillator):
        """some txt"""
        raise NotImplementedError("Please Implement this method in child class")


class CentralOscillator(Oscillator):

    def __init__(self, freq, phase, alpha, beta):
        super().__init__(freq, phase)
        self.alpha = alpha
        self.beta = beta

    def step(self, periferal_osc: list[list[Oscillator]]):
        """some txt"""
        
        pass


class PeripheralOscillator(Oscillator):

    def __init__(self, freq, phase, alpha):
        super().__init__(freq, phase)
        self.alpha = alpha

    def step(self, central_oscillator: CentralOscillator):
        """some txt"""
        pass

    def get_synchonization_state(self, central_oscillator: CentralOscillator):
        """some txt"""
        
        pass