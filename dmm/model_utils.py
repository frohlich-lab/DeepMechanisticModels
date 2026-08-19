def generate_layer_sizes(
        latent_dim: int,
        depth: int,
        max_width: int,
        multiplier: int = 2,  # same default value as conf.nn_structure_multiplier
        reverse: bool = False,
        smoother_transition: bool = True,  # can disable to go back to previous behaviour
):
    layer_sizes = []

    # Determine layer sizes via successive multiplications of bottleneck size
    # The resulting network architecture applies to a decoder/inflater module
    size = multiplier * latent_dim
    while len(layer_sizes) < depth and size <= max_width:
        layer_sizes.append(size)
        if smoother_transition:
            # If the hidden layer closest to the input (encoder) / output (inflater) is too narrow compared to the
            # said input/output, increase the multiplier to bridge the gap (subject to max multiplier value)
            if (len(layer_sizes) == depth - 1) and (size * (multiplier+1) < max_width):
                multiplier = min((max_width/multiplier) / size, 5)
        size = round(size * multiplier)

    # If the max width is hit, pad the rest of the layers with encoder_max_width
    while len(layer_sizes) < depth:
        layer_sizes.append(max_width)

    # Reverse layer size order for an encoder module
    if reverse:
        layer_sizes.reverse()

    return layer_sizes
