import numpy as np
from typing import List
from source.utils import are_close

class Oscillator():

    def __init__(self, freq, phase):
        """some txt"""
        self.freq:float = freq
        self.phase:float = phase
    
    def step(self, oscillator):
        """some txt"""
        raise NotImplementedError("Please Implement this method in child class")


class CentralOscillatorSelAtt(Oscillator):

    def __init__(self, freq, phase, params: list):
        super().__init__(freq, phase)
        self.params = params

    def step(self, periferal_osc: list[np.ndarray[Oscillator]]):
        """some txt"""
        delta_phase = self.freq
        for i, ensemble_po in enumerate(periferal_osc):   
            partial_sum = np.sum(list(map(lambda x: np.sin(x.phase - self.phase),
                                           ensemble_po)))
            delta_phase += self.params[i] * partial_sum / len(ensemble_po)
        self.phase += delta_phase

class CentralOscillatorSegmentation(Oscillator):

    def __init__(self, freq, phase):
        super().__init__(freq, phase)
        self.delta_phase: float
        self.delta_freq: float

    def step(self, s_area_osc: np.ndarray[Oscillator], w1:float, alpha:float, g:callable):
        """some txt"""
        vectorized_g = np.vectorize(g)
        delta_phase = self.freq + w1 / len(s_area_osc) * np.sum(vectorized_g(s_area_osc.phase - self.phase))
        self.delta_phase = delta_phase

        delta_freq = alpha * (self.delta_phase - self.freq)
        self.delta_freq = delta_freq

    def update(self):
        self.phase += self.delta_phase
        self.freq += self.delta_freq

class PeripheralOscillatorSelAtt(Oscillator):

    def __init__(self, freq, phase, alpha):
        super().__init__(freq, phase)
        self.alpha = alpha

    def step(self, central_oscillator: CentralOscillatorSelAtt):
        """some txt"""
        self.phase += self.freq + self.alpha * np.sin(central_oscillator.phase - self.phase)

    def get_synchonization_state(self, central_oscillator: CentralOscillatorSelAtt):
        """some txt"""
        tol = 100
        if are_close(self.phase, central_oscillator.phase, abs_tol=tol):
            return 1
        else:
            return 0
        
class PeripheralOscillatorSegmentation(Oscillator):

    def __init__(self, freq, phase, level, state:int = 1):
        """some txt
        Args:
            state: 1 is active oscillator, 0 is disabled (border_ones)"""
        super().__init__(freq, phase)
        self.state = state
        self.level = level
        self.delta_phase: float
        
    def disable(self):
        """some txt"""
        self.state = 0
        return self

    def get_synchonization_state(self, central_oscillator: CentralOscillatorSegmentation):
        """some txt"""
        pass

    def update(self):
        self.phase += self.delta_phase

class PeripheralOscillatorSegmentationL1(PeripheralOscillatorSegmentation):

    def __init__(self, freq, phase, level, state: int = 1):
        super().__init__(freq, phase, level, state)

    def step(self, 
             central_oscillator: CentralOscillatorSegmentation, 
             w2:callable, 
             w3:callable, 
             point:list, 
             s_size:int, 
             s_start_p:list,
             neibours:List[List[PeripheralOscillatorSegmentation, list]],
             t:int):
        """some txt"""
        s_i = 0
        i, j = point
        if i > s_start_p[0] and i < s_start_p[0] + s_size and \
           j > s_start_p[1] and j < s_start_p[1] + s_size:
            s_i = 1

        delta_phase = self.phase - s_i * w2(t)*np.sin(central_oscillator.phase - self.phase) + \
                      w3(t) / len(neibours) * sum(list(map(lambda x: np.sin(x[0].phase - self.phase), neibours)))
        self.delta_phase = delta_phase   
             
class PeripheralOscillatorSegmentationL2(PeripheralOscillatorSegmentation):

    def __init__(self, freq, phase, level, state: int = 1):
        super().__init__(freq, phase, level, state)
        self.delta_freq:float

    def step(self,
             w4:callable,
             neibours:List[List[PeripheralOscillatorSegmentation, list]],
             senders_to_l2:List[List[PeripheralOscillatorSegmentation, list]],
             w5:callable,
             point:list,
             beta:float,
             t:int):
        
        delta_phase = self.freq + w4(t) / len(neibours) * sum(list(map(lambda x: np.sin(x[0].phase - self.phase), neibours))) + \
                      sum(list(map(lambda x: w5(point,x[1]) * np.sin(x[0].phase - self.phase), senders_to_l2))) / len(senders_to_l2)
        self.delta_phase = delta_phase

        delta_freq = beta * (self.delta_phase - self.freq)
        self.delta_freq = delta_freq

    def update(self):
        self.phase += self.delta_phase
        self.freq += self.delta_freq