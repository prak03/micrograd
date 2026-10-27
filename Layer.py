from Neuron import Neuron
class Layer:
    def __init__(self, layer_size, previous_layer_size, learning_rate):
        self.Neurons = [Neuron(previous_layer_size, learning_rate) for layers in range(layer_size)]

    def forward_call(self, inputs, final_layer):
        updated_inputs = []
        for Neuron in self.Neurons:
            updated_inputs.append(Neuron.forward_pass(inputs, final_layer))
        return updated_inputs

    def update_weights_and_biases(self):
        for Neuron in self.Neurons:
            Neuron.update_weights_and_biases()
