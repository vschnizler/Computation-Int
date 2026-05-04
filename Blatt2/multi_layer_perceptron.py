import numpy as np
from typing import Callable, List

def vector_dot(x: np.ndarray, w: np.ndarray):
    return np.dot(x,w)

def fermi_dirac(x: float):
    return 1/ (1+ np.exp(-x))

def gen_weight():
    return float(2*(np.random.random() - 0.5))

class Neuron:
    def __init__(self,
               summation_func: Callable[[np.ndarray, np.ndarray], float],
               transfer_func: Callable[[float], float]
        ) -> None:
        self.summation_fn: Callable = summation_func
        self.transfer_fn: Callable  = transfer_func
        self.net: float        = 0.0
        self.activation: float = 0.0
        self.delta: float = 0.0

    def compute(self, inputs: np.ndarray, weights: np.ndarray):
        
        self.net = self.summation_fn(inputs, weights)
        self.activation = self.transfer_fn(self.net)
        
        return self.activation

class MultiLayerPerceptron:
    def __init__(self,
        network_structure: np.ndarray,
        input_shape: int,
        output_shape: int,
        summation_func: Callable[[np.ndarray, np.ndarray], float],
        transfer_func: Callable[[float], float]
        ) -> None:
        summation_func: Callable[[np.ndarray, np.ndarray], float]
        transfer_func: Callable[[float], float]
        
        self.neuron_structure: list[list[Neuron]] = [
            [Neuron(summation_func, transfer_func) for neuron_index in range(0,network_structure[layer_index])]
            for layer_index in range(0, len(network_structure)-1)
        ]
        self.matrices: list[list[np.ndarray[float]]] = [
            [
            [gen_weight() for prev_neuron in self.neuron_structure[layer_index-1]]
            for neuron_index in self.neuron_structure[layer_index]]
            for layer_index in range(1, len(self.neuron_structure))
        ]
    
    

    
test = MultiLayerPerceptron(
    [1, 20, 1],
    1,
    1,
    vector_dot,
    fermi_dirac
)

    
    
        
    
    