import numpy as np
from typing import Callable, List


# Dot Produt
def vector_dot(x: np.ndarray, w: np.ndarray):
    return np.dot(x,w)

# Non-linear transfer function
def fermi_dirac(x: float):
    return 1/ (1+ np.exp(-x))

# Generates a weight from (-1,1)
def gen_weight():
    return float(2*(np.random.random() - 0.5))

# Target to be learned
def target_function(x):
    return(
        np.sin(np.abs(x) + 0.5) - 3*np.cos(-x) + 0.7*x
    )

# Square Loss
def loss_func(t,y):
    return(
        0.5 * (t-y)**2
    )

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

    # Computes the output of the Neuron, given an input vector (output of entire previous layer) and a weight vector (unique to each neuron)
    def compute(self, inputs: np.ndarray, weights: np.ndarray):
        
        self.net = self.summation_fn(inputs, weights)
        self.activation = self.transfer_fn(self.net)
        
        return self.activation  # Saves activation for later use in Backprop.

class MultiLayerPerceptron:
    def __init__(self,
        network_structure: np.ndarray, # Build Network of structure [#number of neuron layer[0], #number of neuron layer[1] etc] with number of entries = number of layers
        input_shape: int,       # Dimension of input vector
        output_shape: int,      # Dimension of output vector
        summation_func: Callable[[np.ndarray, np.ndarray], float],
        transfer_func: Callable[[float], float]
        ) -> None:
        summation_func: Callable[[np.ndarray, np.ndarray], float]
        transfer_func: Callable[[float], float]
        self.result: np.ndarray
        
        # Build a List of all layers, used to access neuron output
        self.neuron_structure: list[list[Neuron]] = [
            [Neuron(summation_func, transfer_func) for neuron_index in range(0,network_structure[layer_index]+1)] # +1 for Bias neuron with permanent activation 1
            for layer_index in range(0, len(network_structure))
        ]
        
        # Build a list of matrices for propagation between each layer
        self.matrices = [
            [
            np.array([gen_weight() for prev_neuron in range(len(self.neuron_structure[layer_index-1]))])
            for neuron_index in range(len(self.neuron_structure[layer_index]) -1)]
            for layer_index in range(1, len(self.neuron_structure))
        ]

    #Returns activation vector of a given layer
    def activation_vector(self, layer_index: int):
        activation = []
        for neuron in (self.neuron_structure[layer_index]):
            activation.append(float(neuron.activation)) 
        activation[-1] = 1.0 # Sets activation of bias neuron to 1
        return np.array(activation)
    
    # Computes Network output for a given input vector
    # Sets activation for all layers
    def forward_run(self, input: np.ndarray):
        
        if(len(input) != len(self.neuron_structure[0]) - 1):
            print("Input vector and input neuron mismatch ", len(input), "  and   ", len(self.neuron_structure[0])-1)
            return
        
        for index, value in enumerate(input):
            print(value)
            self.neuron_structure[0][index].activation = float(value) # Sets activation of dummy input neuron to the input
        
        self.neuron_structure[0][-1].activation = 1 # Sets activation of bias neuron to 1
        
        for layer_index in range(1, len(self.neuron_structure)):        # Start at layer[1] to skip dummy input layer
            for neuron_index in range(0, len(self.matrices[layer_index-1])):    # Skip final (bias) neuron
                prev_activation = self.activation_vector(layer_index-1)     # Input of Layer[i] ist the output of layer[i-1]. This expicitly includes bias neuron of prev layer
                
                self.neuron_structure[layer_index][neuron_index].compute(
                    prev_activation,
                    self.matrices[layer_index-1][neuron_index]      # Matrix between layer[i-1] and layer[i] gives input of layer[i]. Skips bias neuron of current layer, since activation is always 1
                )
        self.result = self.activation_vector(-1)[:-1]            # Activation of final layer gives result. [:-1] cutts of bias neuron in output
        
test = MultiLayerPerceptron(
    [1, 20, 1],
    1,
    1,
    vector_dot,
    fermi_dirac
)

training_data = target_function(np.linspace(-10, 10, 1001))

for data in training_data:
    test.forward_run(np.array([data]))


    
    
        
    
    