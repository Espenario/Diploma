import numpy as np
from typing import List, Tuple
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

    def calculate_step_phase(self, phase, s_area_osc: np.ndarray[Oscillator], w1:float, g:callable):
        delta_phase = self.freq + w1 / len(s_area_osc) * sum(list(map(lambda x: g(x.phase - phase), s_area_osc)))
        return delta_phase
    
    def calculate_step_freq(self, freq, alpha):
        return alpha * (self.delta_phase - freq)

    def step(self, s_area_osc: np.ndarray[Oscillator], w1:float, alpha:float, g:callable):
        """some txt""" 
        delta_t = 1

        k1 = self.calculate_step_phase(self.phase, s_area_osc, w1, g)
        k2 = self.calculate_step_phase(self.phase + k1*delta_t/2, s_area_osc, w1, g)
        k3 = self.calculate_step_phase(self.phase + k2*delta_t/2, s_area_osc, w1, g)
        k4 = self.calculate_step_phase(self.phase + k3*delta_t, s_area_osc, w1, g)
        self.delta_phase = self.phase + (delta_t / 6) * (k1 + 2*k2 + 2*k3 + k4)

        k1_f = self.calculate_step_freq(self.freq, alpha=alpha)
        k2_f = self.calculate_step_freq(self.freq + k1_f*delta_t/2, alpha=alpha)
        k3_f = self.calculate_step_freq(self.freq + k2_f*delta_t/2, alpha=alpha)
        k4_f = self.calculate_step_freq(self.freq + k3_f*delta_t, alpha=alpha)
        delta_freq = self.freq + (delta_t / 6) * (k1_f + 2*k2_f + 2*k3_f + k4_f)
        self.delta_freq = delta_freq

    def update(self):
        self.phase += self.delta_phase
        self.phase = self.phase // (2 * np.pi)
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
        self.delta_phase: float = 0
        
    def disable(self):
        """some txt"""
        self.state = 0
        return self

    def get_synchonization_state(self, central_oscillator: CentralOscillatorSegmentation):
        """some txt"""
        pass

    def update(self):
        if self.state == 1:
            self.phase += self.delta_phase
            self.phase = self.phase // (2 * np.pi)

class PeripheralOscillatorSegmentationL1(PeripheralOscillatorSegmentation):

    def __init__(self, freq, phase, level, state: int = 1):
        super().__init__(freq, phase, level, state)

    def calculate_step_phase(self, phase, t:int, central_oscillator: CentralOscillatorSegmentation, 
                                w2:callable, 
                                w3:callable, 
                                w2_alpha, w2_beta, w3_alpha, w3_beta,
                                point:list, 
                                s_size:int, 
                                s_start_p:list,
                                neibours:List[List[Tuple[PeripheralOscillatorSegmentation, list]]],
                               ):
        
        s_i = 1
        eps = 1e-6
        i, j = point
        if i > s_start_p[0] and i < s_start_p[0] + s_size and \
            j > s_start_p[1] and j < s_start_p[1] + s_size:
            s_i = 0
        
        active_neibours = [x for x in neibours if x[0].state == 1]
        
        delta_phase = phase - s_i * w2(t, w2_alpha, w2_beta)*np.sin(central_oscillator.phase - phase) + \
                    w3(t, w3_alpha, w3_beta) / (len(active_neibours) + eps) * sum(list(map(lambda x: np.sin(x[0].phase - phase), active_neibours)))

        return delta_phase
        

    def step(self, 
             central_oscillator: CentralOscillatorSegmentation, 
             w2:callable, 
             w3:callable, 
             w2_alpha, w2_beta, w3_alpha, w3_beta,
             point:list, 
             s_size:int, 
             s_start_p:list,
             neibours:List[List[Tuple[PeripheralOscillatorSegmentation, list]]],
             t:int):
        """some txt"""
        delta_t = 1

        if self.state == 1:
            k1 = self.calculate_step_phase(self.phase, t, central_oscillator, w2, w3, w2_alpha, 
                                            w2_beta, w3_alpha, w3_beta, point, s_size, s_start_p, neibours)
            k2 = self.calculate_step_phase(self.phase+k1*delta_t/2, t+delta_t/2, central_oscillator, w2, w3, w2_alpha, 
                                            w2_beta, w3_alpha, w3_beta, point, s_size, s_start_p, neibours)
            k3 = self.calculate_step_phase(self.phase+k2*delta_t/2, t+delta_t/2, central_oscillator, w2, w3, w2_alpha, 
                                            w2_beta, w3_alpha, w3_beta, point, s_size, s_start_p, neibours)
            k4 = self.calculate_step_phase(self.phase+k3*delta_t, t+delta_t, central_oscillator, w2, w3, w2_alpha, 
                                            w2_beta, w3_alpha, w3_beta, point, s_size, s_start_p, neibours)
            self.delta_phase = self.phase + (delta_t / 6) * (k1 + 2*k2 + 2*k3 + k4)
             
class PeripheralOscillatorSegmentationL2(PeripheralOscillatorSegmentation):

    def __init__(self, freq, phase, level, state: int = 1):
        super().__init__(freq, phase, level, state)
        self.delta_freq:float

    def calculate_step_phase(self, phase, w4, neibours:List[List[Tuple[PeripheralOscillatorSegmentation, list]]],
             senders_to_l2:List[List[Tuple[PeripheralOscillatorSegmentation, list]]],
             w5:callable,
             point:list):
        active_senders_to_l2 = [x for x in senders_to_l2 if x[0].state == 1]
        
        eps = 1e-6
        neibours_impact = w4 / (len(neibours) + eps) * \
                          sum(list(map(lambda x: np.sin(x[0].phase - phase), neibours)))
        
        l1_osc_impact = sum(list(map(lambda x: w5(point,x[1]) * np.sin(x[0].phase - phase),  
                                     active_senders_to_l2))) / (len(active_senders_to_l2) + eps)
        
        delta_phase = self.freq + neibours_impact + l1_osc_impact
        return delta_phase
    
    def calculate_step_freq(self, freq, alpha):
        return alpha * (self.delta_phase - freq)

    def step(self,
             w4:float,
             neibours:List[List[Tuple[PeripheralOscillatorSegmentation, list]]],
             senders_to_l2:List[List[Tuple[PeripheralOscillatorSegmentation, list]]],
             w5:callable,
             point:list,
             beta:float,
             t:int):
        
        delta_t = 1
        k1 = self.calculate_step_phase(self.phase, w4, neibours, senders_to_l2, w5, point)
        k2 = self.calculate_step_phase(self.phase+k1*delta_t/2, w4, neibours, senders_to_l2, w5, point)
        k3 = self.calculate_step_phase(self.phase+k2*delta_t/2, w4, neibours, senders_to_l2, w5, point)
        k4 = self.calculate_step_phase(self.phase+k3*delta_t, w4, neibours, senders_to_l2, w5, point)
        self.delta_phase = self.phase + (delta_t / 6) * (k1 + 2*k2 + 2*k3 + k4)

        k1_f = self.calculate_step_freq(self.freq, alpha=beta)
        k2_f = self.calculate_step_freq(self.freq + k1_f*delta_t/2, alpha=beta)
        k3_f = self.calculate_step_freq(self.freq + k2_f*delta_t/2, alpha=beta)
        k4_f = self.calculate_step_freq(self.freq + k3_f*delta_t, alpha=beta)
        delta_freq = self.freq + (delta_t / 6) * (k1_f + 2*k2_f + 2*k3_f + k4_f)
        self.delta_freq = delta_freq

    def update(self):
        self.phase += self.delta_phase
        self.phase = self.phase // (2 * np.pi)
        self.freq += self.delta_freq