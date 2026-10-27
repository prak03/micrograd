from Value import Value
import random
class Neuron:
    def __init__(self, previous_layer_size, learning_rate):
        self.weights = [Value(random.uniform(-1, 1)) for s in range(previous_layer_size)]
        self.bias = Value(0)
        self.learning_rate = learning_rate

    def forward_pass(self, inputs, final_layer):
        updated_inputs = []
        sum_val = Value(0)
        for num in range(len(inputs)):
            sum_val+= inputs[num]*self.weights[num]
        sum_val+=self.bias
        if not final_layer:
            sum_val = sum_val.relu()
        return sum_val

    def update_weights_and_biases(self):
        for weight in self.weights:
            weight.value -= self.learning_rate * weight.grad
            weight.grad = 0
        self.bias.value -= self.learning_rate * self.bias.grad
        self.bias.grad = 0