def generate_layer_sizes(
        latent_dim: int,
        depth: int,
        max_width: int,
        multiplier: int = 2,  # same default value as conf.nn_structure_multiplier
        reverse: bool = False,
):
    layer_sizes = []

    # Determine layer sizes via successive multiplications of bottleneck size
    # The resulting network architecture applies to a decoder/inflater module
    size = multiplier * latent_dim
    while len(layer_sizes) < depth and size <= max_width:
        layer_sizes.append(size)
        size *= multiplier

    # If the max width is hit, pad the rest of the layers with encoder_max_width
    while len(layer_sizes) < depth:
        layer_sizes.append(max_width)

    # Reverse layer size order for an encoder module
    if reverse:
        layer_sizes.reverse()

    return layer_sizes


# Utility function to compute the total number of parameters (weights & biases) in a DMM -- not in use
# def get_param_number(model):
#     """
#     Get the number of parameters in a model.
#     """
#     num_params = 0
#     for layer in model.deep_encoder.layers + model.deep_inflater.layers:
#         num_params += layer.weight.shape[0] * layer.weight.shape[1]
#         if model.use_layer_bias:
#             num_params += len(layer.bias)
#     return num_params
