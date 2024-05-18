import numpy as np
from source.utils import are_close

class Oscillator():

    def __init__(self, freq, phase):
        """some txt"""
        self.freq:float = freq
        self.phase:float = phase
    
    def step(self, oscillator):
        """some txt"""
        raise NotImplementedError("Please Implement this method in child class")


class CentralOscillator(Oscillator):

    def __init__(self, freq, phase, params: list):
        super().__init__(freq, phase)
        self.params = params

    def step(self, periferal_osc: list[np.ndarray[Oscillator]]):
        """some txt"""
        delta_phase = self.freq
        for i, ensemble_po in enumerate(periferal_osc):   
            partial_sum = np.sum(list(map(lambda x: np.sin(x.phase - self.phase),
                                           ensemble_po)))
            delta_phase += self.params[i] * partial_sum / len(periferal_osc)
        self.phase += delta_phase


class PeripheralOscillator(Oscillator):

    def __init__(self, freq, phase, alpha):
        super().__init__(freq, phase)
        self.alpha = alpha

    def step(self, central_oscillator: CentralOscillator):
        """some txt"""
        self.phase += self.freq + self.alpha * np.sin(central_oscillator.phase - self.phase)

    def get_synchonization_state(self, central_oscillator: CentralOscillator):
        """some txt"""
        tol = 0.1
        if are_close(self.phase, central_oscillator.phase, rel_tol=tol):
            return 1
        else:
            return 0
