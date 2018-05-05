import numpy as np

def get_zeroes_biases_vectors(input_layer_size=None, hidden_layers_size=None, output_layer_size=None):
    biases = []
    all_layers_size = (input_layer_size,) + hidden_layers_size + (output_layer_size,)
    for i in range(0, len(all_layers_size) - 1):
        biases.append(np.zeros(all_layers_size[i + 1]))

    # print("Successfully generated basic biases vector")
    return biases