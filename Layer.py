from Nueron import Nueron
class Layer:
    def __init__(self, layer_size, previous_layer_size, learning_rate):
        self.nuerons = [Nueron(previous_layer_size, learning_rate) for layers in range(layer_size)]

    def forward_call(self, inputs, final_layer):
        updated_inputs = []
        for nueron in self.nuerons:
            updated_inputs.append(nueron.forward_pass(inputs, final_layer))
        return updated_inputs

    def update_weights_and_biases(self):
        for nueron in self.nuerons:
            nueron.update_weights_and_biases()
