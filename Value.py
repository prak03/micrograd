import math
class Value:
    def __init__(self, value, _op="", origins=(), label=""):
        self.value = value
        self.origins = tuple(origins)
        self.operation = _op
        self.grad = 0.0000
        self.label = label
    def __repr__(self):
        return f"Value: {self.value}"

    def __add__(self,other):
        return Value(self.value + other.value, "+", (self, other))
        
    def __mul__(self,other):
        return Value(self.value * other.value,"*", (self, other))
    
    def __sub__(self, other):
        return Value(self.value - other.value, "-", (self,other))

    def tanh(self):
        t = math.tanh(self.value)
        return Value(t, "tanh", (self,))

    def relu(self):
        t = max(0, self.value)
        return Value(t, "relu", (self,))

    def gradient_finder(self):
        if self.operation == "*":
            a, b = self.origins
            a.grad += self.grad * b.value
            b.grad += self.grad * a.value
        elif self.operation == "+":
            a, b = self.origins
            a.grad += self.grad
            b.grad += self.grad
        elif self.operation == "tanh":
            (a,) = self.origins
            a.grad += (1 - self.value ** 2) * self.grad
        elif self.operation == "-":
            a,b = self.origins
            a.grad+=self.grad
            b.grad-=self.grad
        elif self.operation == "relu":
            (a,) = self.origins
            a.grad += (1 if self.value >0 else 0)* self.grad


    def backward_pass(self):
        self.grad = 1

        #find the topo sort order
        visited = set()
        topo_list = []

        def topo(node):
            if node not in visited:
                visited.add(node)
                for origin in node.origins:
                    topo(origin)
                topo_list.append(node)
        topo(self)
        #find the gradients based on that order
        for top in topo_list[::-1]:
            if top.origins:
                top.gradient_finder()
    
def loss_function(predicted, actual):
    if len(predicted) != len(actual):
        raise ValueError(f"{len(predicted)} predictions but {len(actual)} targets")
    total = Value(0)
    for p, a in zip(predicted, actual):
        diff = p - a
        total = total + diff * diff
    return total
