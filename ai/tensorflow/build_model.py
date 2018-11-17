#!/usr/bin/python3
# This will build an empty base model, with training ops that can be executed from Go.
import tensorflow as tf

tf.app.flags.DEFINE_string("output", "", "Where to save the graph definition.")
FLAGS = tf.app.flags.FLAGS


# Model internal type: tf.float16 presumably is faster in the RX2080 Ti GPU,
# and not slower in others.
MODEL_DTYPE=tf.float16

# Dimension of the input features.
BOARD_FEATURES_DIM = 41  # Should match ai.AllFeaturesDim

# These should match the same in policy_features.go
ACTION_FEATURES_DIM = 1  # Static/context features.
NUM_SECTIONS = 6  # Sections of neighbourhood.
POSITIONS_PER_SECTION = 3  # Num of board positions per section.
FEATURES_PER_POSITION = 16  # Num of features per position.
# NEIGHBOURHOOD_NUM_FEATURES = 305
NEIGHBOURHOOD_NUM_FEATURES = (
        (1 + POSITIONS_PER_SECTION*NUM_SECTIONS) * FEATURES_PER_POSITION + ACTION_FEATURES_DIM)

# Training parameters.

# Neural Network parameters for board embedding and value prediction.
BOARD_NUM_HIDDEN_LAYERS = 3
BOARD_NODES_PER_LAYER = 128 - BOARD_FEATURES_DIM
BOARD_EMBEDDING_DIM = 64 - BOARD_FEATURES_DIM

# Neural Network parameters for actions embedding and value prediction.
NEIGHBOURHOOD_NUM_HIDDEN_LAYERS = 3
NEIGHBOURHOOD_NODES_PER_LAYER = 384 - NEIGHBOURHOOD_NUM_FEATURES
NEIGHBOURHOOD_EMBEDDING_DIM = 32


def SigmoidTo10(x, smoothness=4.0):
	"""Make a sigmoid curve on values > 9.8 or < -9.8."""
	abs_x = tf.abs(x)
	threshold = tf.constant(9.8, dtype=MODEL_DTYPE)
	mask = (abs_x > threshold)
	sigmoid = tf.sigmoid((abs_x - threshold) / smoothness)
	sigmoid = threshold + (sigmoid - 0.5) * 0.4   # 0.4 = 0.2 / 0.5
	sigmoid = tf.sign(x) * sigmoid
	return tf.where(abs_x > threshold, sigmoid, x)


def SparseLogSoftMax(logits, indices):
    with tf.name_scope("SparseLogSoftMax"):
        if len(indices.shape) == 1:
            indices = tf.expand_dims(indices, 1)
        batch_size = tf.math.reduce_max(indices) + 1
        num_values = tf.cast(tf.shape(logits)[0], tf.int64)
        dense_shape_2d = [batch_size, num_values]
        indices_2d = tf.concat([indices,
                                tf.expand_dims(tf.range(num_values), 1)], axis=1)
        sparse_logits = tf.SparseTensor(indices=indices_2d, values=logits, dense_shape=dense_shape_2d)
        logits_max = tf.sparse_reduce_max(sp_input=sparse_logits, axis=-1, keepdims=True)
        logits_max = tf.reshape(tf.manip.gather_nd(logits_max, indices), [-1])
        # Propagating the gradient through logits_max should be a no-op, so to accelrate this
        # we just prune it,
        # Also tf.sparse_reduce_max doesn't have a gradient implemented in TensorFlow (as of Nov/2018).
        logits_max = tf.stop_gradient(logits_max)
        normalized_logits = logits - logits_max
        normalized_exp_values = tf.exp(normalized_logits)
        normalized_exp_sum = tf.manip.scatter_nd(indices, updates=normalized_exp_values, shape=[batch_size])
        normalized_log_exp_sum = tf.manip.gather_nd(params=tf.log(normalized_exp_sum), indices=indices)
        return normalized_logits - normalized_log_exp_sum


def SparseCrossEntropyLoss(log_probs, labels):
    return labels * -log_probs

# Neither num_hidden_layers_nodes and output_embedding_dim include the dimensions of the input
# that may be concatenated for the skip connections.
def buildSkipFFNN(input, num_hidden_layers, num_hidden_layers_nodes,
                  skip_also_output, output_embedding_dim, initializer, l2_regularizer):
    with tf.name_scope("buildSkipFFNN"):
        logits = input
        if num_hidden_layers > 0:
            for ii in range(num_hidden_layers-1):
                logits = tf.layers.dense(logits, num_hidden_layers_nodes, tf.nn.selu,
                                         kernel_initializer=initializer, kernel_regularizer=l2_regularizer,
                                         name="hidden_{}".format(ii), reuse=tf.AUTO_REUSE)
                logits = tf.concat([logits, input], 1)
            # Last hidden layer can be of different size, and the skip connection is optional.
            logits = tf.layers.dense(logits, output_embedding_dim, tf.nn.selu,
                                     kernel_initializer=initializer, kernel_regularizer=l2_regularizer,
                                     name="embedding", reuse=tf.AUTO_REUSE)
            if skip_also_output:
                logits = tf.concat([logits, input], 1)
    return logits


def BuildBoardEmbeddings(board_features, initializer, l2_regularizer):
    with tf.name_scope("BuildBoardEmbeddings"):
        board_features = tf.cast(board_features, MODEL_DTYPE)
        with tf.variable_scope("board_kernel", reuse=tf.AUTO_REUSE):
                logits = buildSkipFFNN(board_features, BOARD_NUM_HIDDEN_LAYERS, BOARD_NODES_PER_LAYER,
                                       True, BOARD_EMBEDDING_DIM, initializer, l2_regularizer)
    return logits


def BuildBoardModel(board_embeddings, board_labels, initializer, l2_regularizer):
    with tf.name_scope("BuildBoardModel"):
        with tf.variable_scope("board_kernel"):
            board_values = tf.layers.dense(board_embeddings, 1, activation=None,
                                           name="linear_layer", kernel_initializer=initializer,
                                           kernel_regularizer=l2_regularizer)
        # Adjust prediction.
        board_predictions = SigmoidTo10(board_values)
        board_labels = tf.cast(board_labels, MODEL_DTYPE)
        reshaped_labels = tf.reshape(board_labels, [-1, 1])
        board_losses = tf.losses.mean_squared_error(reshaped_labels, board_values,
                                                    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        board_losses= tf.cast(board_losses, MODEL_DTYPE)
        return (board_predictions, board_losses)


def BuildNeighbourhoodEmbeddings(actions_features, center, neighbourhood, initializer, l2_regularizer):
    with tf.name_scope("BuildNeighbourhoodEmbeddings"):
        rotation_embeddings = []
        with tf.variable_scope("neighbourhood_kernel", reuse=tf.AUTO_REUSE):
            for rotation in range(6):
                neigh_rotated = neighbourhood
                if rotation > 1:
                    neigh_rotated = tf.manip.roll(neighbourhood, shift=rotation, axis=1)
                neigh_concated = tf.reshape(neigh_rotated, [
                    tf.shape(neigh_rotated)[0], neighbourhood.shape[1] * neighbourhood.shape[2]])
                all = tf.concat([actions_features, center, neigh_concated], axis=1)
                embedding = buildSkipFFNN(all, NEIGHBOURHOOD_NUM_HIDDEN_LAYERS, NEIGHBOURHOOD_NODES_PER_LAYER,
                                          False, NEIGHBOURHOOD_EMBEDDING_DIM, initializer, l2_regularizer)
                rotation_embeddings.append(embedding)
        all_embeddings = tf.stack(rotation_embeddings, axis=1)
        sum = tf.reduce_sum(all_embeddings, axis=1)
        max = tf.reduce_max(all_embeddings, axis=1)
        return tf.concat([sum, max], axis=1)


def BuildActionsModel(board_embeddings,
                           actions_board_indices, actions_features,
                           actions_source_center, actions_source_neighbourhood,
                           actions_target_center, actions_target_neighbourhood,
                           actions_labels, initializer, l2_regularizer):
    with tf.name_scope("BuildActionsModel"):
        actions_features = tf.cast(actions_features, MODEL_DTYPE, name="cast_actions_features")
        actions_source_center = tf.cast(actions_source_center, MODEL_DTYPE, name="cast_actions_source_center")
        actions_source_neighbourhood = tf.cast(actions_source_neighbourhood, MODEL_DTYPE,
                                               name="cast_actions_source_neighbourhood")
        actions_target_center = tf.cast(actions_target_center, MODEL_DTYPE, name="cast_actions_target_center")
        actions_target_neighbourhood = tf.cast(actions_target_neighbourhood, MODEL_DTYPE,
                                               name="cast_actions_target_neighbourhood")
        actions_board_indices = tf.cast(actions_board_indices, tf.int64, name="cast_actions_board_indices")
        actions_labels = tf.cast(actions_labels, MODEL_DTYPE)

        # Broadcast board_embeddings to each action.
        broadcasted_board_embeddings = tf.manip.gather_nd(
            params=board_embeddings, indices=tf.expand_dims(actions_board_indices, axis=-1))
        # gather_nd loses the last dimension (it should keep it):
        broadcasted_board_embeddings = tf.reshape(broadcasted_board_embeddings, [-1, board_embeddings.shape[1]])

        # Build embeddings from neighbourhoods.
        with tf.variable_scope("source_pos", reuse=tf.AUTO_REUSE):
            source_embedding = BuildNeighbourhoodEmbeddings(
                actions_features, actions_source_center, actions_source_neighbourhood, initializer, l2_regularizer)
            is_move = actions_features[:,0] > 0
            source_embedding = tf.where(
                is_move,
                source_embedding,
                tf.zeros_like(source_embedding))
        with tf.variable_scope("target_pos", reuse=tf.AUTO_REUSE):
            target_embedding = BuildNeighbourhoodEmbeddings(
                actions_features, actions_target_center, actions_target_neighbourhood, initializer, l2_regularizer)

        # Put all features together.
        actions_logits = tf.concat([actions_features, broadcasted_board_embeddings,
                                    source_embedding, target_embedding], axis=1)

        # Build loss and predictions.
        with tf.variable_scope("actions_kernel"):
            actions_logits = tf.layers.dense(inputs=actions_logits, units=1, activation=None, name="linear_layer",
                                             kernel_initializer=initializer, kernel_regularizer=l2_regularizer)
        log_soft_max = SparseLogSoftMax(tf.reshape(actions_logits,[-1]), actions_board_indices)
        actions_predictions = tf.exp(log_soft_max)
        actions_loss = tf.reduce_sum(SparseCrossEntropyLoss(log_soft_max, actions_labels))
        return (actions_predictions, actions_loss)

# Saves graph and returns a SaverDef that can be used to save checkpoints.
def SaveGraph(output_path):
    with open(output_path, 'wb') as f:
        f.write(tf.get_default_graph().as_graph_def().SerializeToString())
    return tf.train.Saver().as_saver_def()


def CreateSaveDef():
    saver_def = tf.train.Saver().as_saver_def()
    print('Feed this tensor to set the checkpoint filename: ',
          saver_def.filename_tensor_name)
    print('Run this operation to save a checkpoint        : ',
          saver_def.save_tensor_name)
    print('Run this operation to restore a checkpoint     : ',
          saver_def.restore_op_name)


def BuildRegularizer(l2):
    l2 = tf.cast(l2, MODEL_DTYPE)
    raw_l2_regularizer = tf.keras.regularizers.l2(1.0)
    return lambda x: raw_l2_regularizer(x) * l2

def main(argv=None):  # pylint: disable=unused-argument
    assert FLAGS.output

    # We are using tf.nn.selu, which requires a special initializer.
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')

    # Build board inputs
    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    l2_regularization = tf.placeholder(tf.float32, shape=(), name='l2_regularization')
    l2_regularizer = BuildRegularizer(l2_regularization)
    board_features = tf.placeholder(tf.float32, shape=[None, BOARD_FEATURES_DIM], name='board_features')
    board_labels = tf.placeholder(tf.float32, shape=[None], name='board_labels')
    print('Board inputs:')
    input_names = [x.name for x in (learning_rate, board_features, board_labels)]
    print('\t{}\n'.format("\t".join(input_names)))

    # Build board logits and model.
    board_embeddings = BuildBoardEmbeddings(board_features, initializer, l2_regularizer)
    board_predictions, board_losses = BuildBoardModel(board_embeddings, board_labels, initializer, l2_regularizer)
    total_losses = board_losses
    board_predictions = tf.identity(
        tf.cast(tf.reshape(board_predictions, [-1]), tf.float32), name='board_predictions')
    board_losses = tf.identity(tf.cast(board_losses, tf.float32), name='board_losses')
    print('Board outputs:')
    print('\t{}\t{}\n'.format(board_predictions.name, board_losses.name))

    # Build per action inputs.
    # All inputs are sparse, since the number of actions is variable.
    # The input `actions_board_indices` list for each actions_features what is the
    # corresponding board -- so the indices are from 0 to len(board_features)-1.
    actions_board_indices = tf.placeholder(tf.int64, shape=[None], name='actions_board_indices')
    actions_features = tf.placeholder(tf.float32, shape=[None, 1], name='actions_features')
    actions_source_center = tf.placeholder(tf.float32, shape=[None, FEATURES_PER_POSITION],
                                           name='actions_source_center')
    actions_source_neighbourhood = tf.placeholder(
        tf.float32, shape=[None, NUM_SECTIONS, POSITIONS_PER_SECTION * FEATURES_PER_POSITION],
        name="actions_source_neighbourhood")
    actions_target_center = tf.placeholder(tf.float32, shape=[None, FEATURES_PER_POSITION],
                                           name='actions_target_center')
    actions_target_neighbourhood = tf.placeholder(
        tf.float32, shape=[None, NUM_SECTIONS, POSITIONS_PER_SECTION * FEATURES_PER_POSITION],
        name="actions_target_neighbourhood")
    actions_labels = tf.placeholder(tf.float32, shape=[None], name='actions_labels')

    print('Action Inputs:')
    input_names = ["{}: {}".format(x.name, x.dtype) for x in (actions_board_indices, actions_features,
                                    actions_source_center, actions_source_neighbourhood,
                                    actions_target_center, actions_target_neighbourhood,
                                    actions_labels)]
    print('\t{}\n'.format('\n\t'.join(input_names)))

    # Build actions model.
    actions_predictions, actions_losses = BuildActionsModel(
        board_embeddings, actions_board_indices, actions_features,
        actions_source_center, actions_source_neighbourhood,
        actions_target_center, actions_target_neighbourhood,
        actions_labels, initializer, l2_regularizer)
    total_losses += actions_losses
    actions_predictions = tf.identity(
        tf.cast(tf.reshape(actions_predictions, [-1]), tf.float32), name='actions_predictions')
    actions_losses = tf.identity(tf.cast(actions_losses, tf.float32), name='actions_losses')
    print('Actions outputs:')
    print('\t{}'.format(actions_predictions.name))
    print('\t{}'.format(actions_losses.name))

    # Build optimizer and train opt.
    global_step = tf.train.create_global_step()
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_losses, global_step=global_step, name='train')
    init = tf.global_variables_initializer()
    print('Training:')
    print('\tInitialize variables:\t', init.name)
    print('\tTrain one step:\t', train_op.name)

    # Mean loss: more stable across batches of different sizes.
    mean_loss = total_losses / tf.cast(tf.shape(board_features)[0], dtype=tf.float16)
    mean_loss = tf.identity(tf.cast(mean_loss, tf.float32), name='mean_loss')
    print('\tMean total loss:\t', mean_loss.name)

    # Create saver nodes.
    CreateSaveDef()

    # Save model
    SaveGraph(FLAGS.output)
    print('Saved to {}'.format(FLAGS.output))

if __name__ == '__main__':
    tf.app.run()
