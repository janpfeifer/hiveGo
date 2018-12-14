#!/usr/bin/python3
import tensorflow as tf

# Predictions should be limited to MAX_Q_VALUE. Values above MAX_Q_LINEAR_VALUE are passed by a sigmoid
# in order to be limited to MAX_Q_VALUE.
MAX_Q_VALUE = 10.0
MAX_Q_LINEAR_VALUE = 9.8


def sigmoid_to_max(x, absolute_max=MAX_Q_VALUE, linear_threshold=MAX_Q_LINEAR_VALUE, smoothness=4.0):
    """Make a sigmoid curve on values > MAX_LINEAR_VALUE or < -MAX_LINEAR_VALUE."""
    abs_x = tf.abs(x)
    threshold = tf.constant(linear_threshold, x.dtype)
    mask = (abs_x > threshold)
    sigmoid = tf.sigmoid((abs_x - threshold) / smoothness)
    sigmoid = threshold + (sigmoid - 0.5) * 2 * (absolute_max - threshold)
    sigmoid = tf.sign(x) * sigmoid
    return tf.where(mask, sigmoid, x)


def sparse_log_soft_max(logits, indices):
    """Sparse version of a `log(softmax)` function.

    There are `BATCH_SIZE` entries, each with an arbitrary number of logits. For
    each value in `logits` there is a corresponding value in `indices` from `0` to `BATCH_SIZE-1`
    which indicate which entry the logit is participating.

    Args:
        logits: dense collection of logits, a concatenation of logits for all entries, of shape `[N]`.
        indices: indices of values `0` to `BATCH_SIZE-1`, that indicates which entry the
            corresponding logit participate. Shape `[N]`.

    Returns `log(softmax)` of the logits, such that the sum of all `exp(values)`
        for the same entry sums to 1. So the `log(probabilities)` depending on how their values are trained.
        A Tensor of shape `[N]`,
    """
    with tf.name_scope("SparseLogSoftMax"):
        if len(indices.shape) == 1:
            indices = tf.expand_dims(indices, 1)
        batch_size = tf.math.reduce_max(indices) + 1
        num_values = tf.cast(tf.shape(logits)[0], tf.int64)
        dense_shape_2d = [batch_size, num_values]
        indices_2d = tf.concat([indices,
                                tf.expand_dims(tf.range(num_values), 1)], axis=1)
        sparse_logits = tf.SparseTensor(
            indices=indices_2d, values=logits, dense_shape=dense_shape_2d)
        logits_max = tf.sparse_reduce_max(
            sp_input=sparse_logits, axis=-1, keepdims=True)
        logits_max = tf.reshape(tf.manip.gather_nd(logits_max, indices), [-1])
        # Propagating the gradient through logits_max should be a no-op, so to accelrate this
        # we just prune it,
        # Also tf.sparse_reduce_max doesn't have a gradient implemented in TensorFlow (as of Nov/2018).
        logits_max = tf.stop_gradient(logits_max)
        normalized_logits = logits - logits_max
        normalized_exp_values = tf.exp(normalized_logits)
        normalized_exp_sum = tf.manip.scatter_nd(
            indices, updates=normalized_exp_values, shape=[batch_size])
        normalized_log_exp_sum = tf.manip.gather_nd(
            params=tf.log(normalized_exp_sum), indices=indices)
        return normalized_logits - normalized_log_exp_sum


def sparse_cross_entropy_loss(log_probs, labels):
    """Log-loss, using as input the output of sparse_log_soft_max."""
    return labels * -log_probs


# Neither num_hidden_layers_nodes and output_embedding_dim include the dimensions of the input
# that may be concatenated for the skip connections.
def build_skip_ffnn(input, num_hidden_layers, num_hidden_layers_nodes,
                    skip_also_output, output_embedding_dim,
                    activation, initializer, l2_regularizer):
    """Builds FFNN with skip connections."""
    with tf.name_scope("buildSkipFFNN"):
        logits = input
        if num_hidden_layers > 0:
            for ii in range(num_hidden_layers - 1):
                with tf.variable_scope("hidden_{}".format(ii), reuse=tf.AUTO_REUSE):
                    logits = tf.layers.dense(logits, num_hidden_layers_nodes, activation,
                                             kernel_initializer=initializer, kernel_regularizer=l2_regularizer,
                                             name="linear", reuse=tf.AUTO_REUSE)
                logits = tf.concat([logits, input], 1)
            # Last hidden layer can be of different size, and the skip connection is optional.
            with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
                logits = tf.layers.dense(logits, output_embedding_dim, activation,
                                         kernel_initializer=initializer, kernel_regularizer=l2_regularizer,
                                         name="linear", reuse=tf.AUTO_REUSE)
            if skip_also_output:
                logits = tf.concat([logits, input], 1)
    return logits


_HEX_SIDES_NAMES = ("left", "right")


def hexagonal_filters(in_channels, out_channels, dtype, initializer=None):
    """Returns 2 separate 3x3 filters: one for even columns one for odd columns.

    Create one underlying set of variables that are composed in two different
    filters of shape `[3, 3, in_channels, out_channels]`.

    This is needed because hexagons neighbourhood, when stored in the form of
    2D matrices, are slightly different for even and odd columns.

    Args:
       in_channels: Depth, or number of channels on the input matrix.
       out_channels: Depth, or number of channels in the output matrix,
                     when applying the filter in a convolution.
       dtype: Value type used in the filters.
       initializer: Variable initializer, or a list of 3 values, one for
         each filter component (center, left and right), of shapes
         `[3, 1, in_channels, out_channels]`, `[2, 1, in_channels, out_channels]`
         and `[2, 1, in_channels, out_channels]` respectivelly.

    Returns:
      List of 2 filters of dimensions `[3, 3, in_channels, out_channels]`
      that are backed up by the same variables, so their values are tied
      together.
    """
    inits = [initializer, initializer, initializer]
    shapes = [
        [3, 1, in_channels, out_channels],
        [2, 1, in_channels, out_channels],
        [2, 1, in_channels, out_channels],
    ]
    if isinstance(initializer, list) or isinstance(initializer, tuple):
        inits = [v for v in initializer]
        shapes = [None, None, None]

    # Filters will be composed of 3 components (columns). The middle one has
    # 3 elemens, and the left and right ones have 2: so 6 neighbours, plus one
    # for the center of the hexagon.
    center = tf.get_variable(
        name="hex_filter_center",
        shape=shapes[0],
        dtype=dtype,
        initializer=inits[0],
    )
    sides = [
        tf.get_variable(
            name="hex_filter_" + _HEX_SIDES_NAMES[ii],
            shape=shapes[ii + 1],
            dtype=dtype,
            initializer=inits[ii + 1],
        )
        for ii in range(2)
    ]

    filters = []
    padding = tf.zeros([1, 1, in_channels, out_channels], dtype=dtype)
    for idx in range(2):
        if idx == 0:
            full_sides = [
                tf.concat([side, padding], axis=0)
                for side in sides
            ]
        else:
            full_sides = [
                tf.concat([padding, side], axis=0)
                for side in sides
            ]
        filters.append(tf.concat([full_sides[0], center, full_sides[1]], axis=1))
    return filters


def hexagonal_conv2d(hex_input, out_channels, filter_initializer=None):
    """Computes convolution(s) of radius one (center+6 neighbours)

    Args:
      input: Tensor of shape `[BATCH_SIZE, HEIGHT, WIDTH, DEPTH]`,
        at least DEPTH must be static (known in graph building time).
      out_channels: Depth, or number of outputs
      filter_initializer: Initializer function to use for filter variables.

    Returns:
      A tensor of shape `[BATCH_SIZE, HEIGHT, WIDTH, out_channels]`.
    """
    with tf.name_scope("hexagonal_conv2d"):
        in_channels = hex_input.shape[3]
        hex_filters = hexagonal_filters(in_channels, out_channels, dtype=hex_input.dtype,
                                        initializer=filter_initializer)
        even_out = tf.nn.conv2d(hex_input, hex_filters[0], strides=[1, 1, 1, 1],
                                padding='SAME', name="even")
        odd_out = tf.nn.conv2d(hex_input, hex_filters[1], strides=[1, 1, 1, 1],
                               padding='SAME', name="odd")
        hex_input_shape = tf.shape(hex_input)
        width = hex_input_shape[2]
        selection_mask = tf.range(width)
        selection_mask = tf.equal(selection_mask % 2, 0)
        selection_mask = tf.reshape(selection_mask, [1, 1, width, 1])
        selection_mask = tf.broadcast_to(selection_mask,
                                         [hex_input_shape[0], hex_input_shape[1], hex_input_shape[2], out_channels])
        mix = tf.where(selection_mask, even_out, odd_out, name="mixed")
    return mix


def hexagonal_layer(hexagonal_map, activation, out_channels=None, initializer=None):
    """Pass hexagonal map by a convolution, activation and sum the input (residual).

    Args:
        hexagonal_map: Input hexagonal map, a tensor of shape `[BATCH_SIZE?, HEIGHT?, WIDTH?, DEPTH]`.
        activation: Activation function, which may include batch normalization.
        out_channels: Number of output channels (depth) in the next layer. If None, it will be set to the same
          of the input. If output channels (depth) is equal or larger to input, the input is added as residual.
        initializer: Passed to the convolution filter creation.
    """
    with tf.name_scope("hexagonal_layer"):
        in_channels = hexagonal_map.shape[3]
        if out_channels is None:
            out_channels = in_channels
        if initializer is None:
            initializer = tf.initializers.truncated_normal(0.0, 1. / (7. * in_channels.value))
        # Output channels == input channels (the depth)
        hexagonal_output = hexagonal_conv2d(hexagonal_map, out_channels, filter_initializer=initializer)
        hexagonal_output = activation(hexagonal_output)
        if in_channels == out_channels:
            # Straight residual connection.
            hexagonal_output = hexagonal_output + hexagonal_map
        elif out_channels > in_channels:
            # Residual connection in the first in_channels values of the embedding.
            input_shape = tf.shape(hexagonal_map)
            padding = tf.zeros(shape=[input_shape[0], input_shape[1], input_shape[2], out_channels - in_channels],
                               dtype=hexagonal_map.dtype)
            hexagonal_output = hexagonal_output + tf.concat([hexagonal_map, padding], axis=3)
        return hexagonal_output
