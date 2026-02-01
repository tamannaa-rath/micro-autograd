import math
class Value:
    def __init__ (self, data, _children=(), _op='', label = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data = {self.data})"

    def __add__(self,other):
        other = other if isinstance (other,Value) else Value(other)
        out = Value(self.data + other.data, (self,other), '+')
        
        def _backward():
            # For addition, the local derivative is 1.0
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out


    def __mul__(self,other):
        other = other if isinstance (other,Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')
        
        def _backward():
            # Chain rule: self.grad is (other.data) * (output's gradient)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        # This implementation only supports int or float powers for now
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
        # Power rule: d/dx(x^n) = n * x^(n-1)
            self.grad += (other * (self.data**(other - 1))) * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other): 
        return self + other
    
    def __rmul__(self,other):
        return self * other

    def __truediv__(self,other):
        return self * (other**-1)

    def __neg__(self):
        return self * -1

    def __sub__(self,other):
        return self + (-other)
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            # 't' is the result of the tanh calculation (self.data after tanh)
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
        

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
    
        return out
        

    def backward(self):
        # Topological sort implementation
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        # Build the ordering starting from the output node 'o'
        build_topo(self)

        # Initialize output gradient
        self.grad = 1.0


        # Automated backward pass through all nodes
        for node in reversed(topo):
            node._backward()

