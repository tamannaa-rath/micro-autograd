import random 
from engine import Value

class Neuron:
    
    def __init__(self, nin):
        # Initialize weights and bias with random values between -1 and 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        # w * x + b
        # sum() uses self.b as the starting value for the summation
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        # Returns a list of all Value objects in this neuron
        return self.w + [self.b]

class Layer:
    
    def __init__(self, nin, nout):
        # Create a list of 'nout' neurons, each taking 'nin' inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        # Pass inputs through every neuron in the layer
        outs = [n(x) for n in self.neurons]
        # Return a single value if there's only one neuron, otherwise a list
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # Returns a flattened list of parameters from all neurons in the layer
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    
    def __init__(self, nin, nouts):
        # Combine input size with the list of output sizes for each layer
        sz = [nin] + nouts
        # Build layers sequentially based on the architecture sizes
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        # Pass the input through each layer in sequence
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        # Returns a flattened list of every parameter in the entire network
        return [p for layer in self.layers for p in layer.parameters()]
