import numpy as np
from typing import Callable, List
import matplotlib.pyplot as plt

# Dot Produt
def vector_dot(x: np.ndarray, w: np.ndarray):
    return np.dot(x,w)

# Non-linear transfer function
def fermi_dirac(x: float):
    return 1/ (1+ np.exp(-x))

def fermi_dirac_derivative(x):
    return(
        np.exp(-x)/((1+np.exp(-x))**2)
    )

def linear_func(x):
    return(
        x
    )

def linear_func_derivative(x):
    return 1.0

# Generates a weight from (-1,1)
def gen_weight():
    return float(2*(np.random.random() - 0.5))

# Target to be learned
def target_function(x):
    return(
        np.sin(15/(np.abs(x) + 0.5)) - 3*np.cos(-x) + 0.7*x
    )

# Square Loss
def loss_func(t,y):
    return(
        0.5 * (t-y)**2
    )

# Derivative of square loss
def loss_func_derviative(t,y):
    return(
        -(t-y)
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
        self.neuron_structure[-1][0].transfer_fn = linear_func
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
    
    
    def backpropagation_final_layer(self, input, target_func, loss_func, transfer_derivative,loss_func_derviative, lerning_rate):
        
        target_values = target_func(input)
        
        
        self.forward_run(input)
        self.calculate_deltas_final_layer(target_values, transfer_derivative, loss_func_derviative)
        self.update_weights_final_layer(lerning_rate)
        
        return loss_func(np.array(target_values), np.array(self.result))
    
    def calculate_deltas_final_layer(self, target_values, transfer_derivative, loss_func_derivative):
        
        for neuron in self.neuron_structure[-1][:-1]:
            t = float(np.squeeze(target_values))
            y = float(np.squeeze(self.result))
            neuron.delta = float(loss_func_derivative(t,y ) * transfer_derivative(neuron.net))
            
    def update_weights_final_layer(self, learning_rate):
        
        for neuron_index, matrixelement in enumerate(self.matrices[-1]):
           
            prev_activation = self.activation_vector(len(self.neuron_structure) - 2)
            
            self.matrices[-1][neuron_index] = matrixelement - learning_rate*self.neuron_structure[-1][neuron_index].delta * prev_activation
            
            
        
test = MultiLayerPerceptron(
    [1, 20, 1],
    1,
    1,
    vector_dot,
    fermi_dirac
)

x = np.linspace(-10, 10, 1001)
errorhist=[]

training_runs = 10000

for run_num in range(training_runs):
    run_hist = []
    for x_scalar in x:
        run_hist.append(test.backpropagation_final_layer(np.array([x_scalar]), target_function, loss_func, linear_func_derivative, loss_func_derviative, 0.001))
    errorhist.append(np.average(run_hist))
plot_yvals = []


plt.plot(np.linspace(0, len(errorhist), len(errorhist)), errorhist)
plt.grid()
plt.show()
for x_scalar in x:
    test.forward_run(np.array([x_scalar]))
    plot_yvals.append(test.result)



plt.plot(x, plot_yvals)
plt.plot(x, target_function(x))
plt.grid()
plt.show()