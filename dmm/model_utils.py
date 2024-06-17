from training_configuration import NN_STRUCTURE_MULTIPLIER


def generate_layer_sizes(
        latent_dim: int,
        depth: int,
        max_width: int,
        multiplier: int = NN_STRUCTURE_MULTIPLIER,
        reverse: bool = False,
):
    layer_sizes = []

    # Determine layer sizes via successive multiplications of bottleneck size
    size = multiplier * latent_dim
    while len(layer_sizes) < depth and size <= max_width:
        layer_sizes.append(size)
        size *= multiplier

    # If the max width is hit, pad the rest of the layers with encoder_max_width
    while len(layer_sizes) < depth:
        layer_sizes.append(max_width)

    if reverse:
        layer_sizes.reverse()

    return layer_sizes
