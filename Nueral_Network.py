import random
from Value import Value
from Layer import Layer
from Value import loss_function
class Nueral_Network:
    def __init__(self, size,  learning_rate=0.008):
        print("lerning rate",learning_rate)
        self.learning_rate = learning_rate
        self.size = size
        self.network = []
        for s in range(1, len(size)):
            self.network.append(Layer(size[s], size[s - 1], learning_rate))

    def forward(self, inputs):
        final_layer = False
        for layer in range(len(self.network)):
            if layer == len(self.network)-1:
                final_layer = True
            inputs = self.network[layer].forward_call(inputs, final_layer)
        return inputs


    def back_pass(self, loss):
        loss.backward_pass()
        self.update_weights_and_biases()
    
    def update_weights_and_biases(self):
        for layer in self.network:
            layer.update_weights_and_biases()

    def count_dead_neurons(self, examples):
        hidden_layers = self.network[:-1]      
        ever_on = [[False] * len(layer.nuerons) for layer in hidden_layers]

        for x, _ in examples:
            inputs = [Value(v) for v in x]
            for l, layer in enumerate(hidden_layers):
                inputs = layer.forward_call(inputs, False)
                for n, out in enumerate(inputs):
                    if out.value > 0:
                        ever_on[l][n] = True

        total_dead = 0
        for l, flags in enumerate(ever_on):
            dead = flags.count(False)
            total_dead += dead
            print(f"layer {l + 1}: {dead} dead out of {len(flags)}")
        total = sum(len(flags) for flags in ever_on)
        print(f"total: {total_dead} dead out of {total}")
        return total_dead
