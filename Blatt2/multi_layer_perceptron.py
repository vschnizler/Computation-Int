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
        self.result: np.ndarray
        self.neuron_structure: list[list[Neuron]] = [
            [Neuron(summation_func, transfer_func) for neuron_index in range(0,network_structure[layer_index]+1)]
            for layer_index in range(0, len(network_structure))
        ]
        self.matrices: list[list[np.ndarray[float]]] = [
            [
            [gen_weight() for prev_neuron in range(len(self.neuron_structure[layer_index-1]))]
            for neuron_index in range(len(self.neuron_structure[layer_index]) -1)]
            for layer_index in range(1, len(self.neuron_structure))
        ]

    def activation_vector(self, layer_index: int):
        activation = []
        for neuron in (self.neuron_structure[layer_index]):
            activation.append(neuron.activation) 
        activation[-1] = 1
        return activation
    
    def forward_run(self, input: np.ndarray):
        
        for index, value in enumerate(input):
            self.neuron_structure[0][index].activation = value
        
        self.neuron_structure[0][-1].activation = 1
        
        for layer_index in range(1, len(self.neuron_structure)):
            for neuron_index in range(0, len(self.matrices[layer_index-1])):
                prev_activation = self.activation_vector(layer_index-1)
                print(prev_activation)
                self.neuron_structure[layer_index][neuron_index].compute(
                    prev_activation,
                    self.matrices[layer_index-1][neuron_index]
                )
        self.result = self.activation_vector(-1)
        
   
        
    
    
    

    
test = MultiLayerPerceptron(
    [1, 20, 1],
    1,
    1,
    vector_dot,
    fermi_dirac
)
test.forward_run(np.array([1]))


    
    
        
    
    