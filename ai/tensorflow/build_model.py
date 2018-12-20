#!/usr/bin/python3
# This will build an empty base model, with training ops that can be executed from Go.
import hive_lib
import tensorflow as tf

# Model internal type: tf.float16 presumably is faster in the RX2080 Ti GPU,
# and not slower in others. Losses are still kept as float32 though.
MODEL_DTYPE = tf.float16


tf.app.flags.DEFINE_string("output", "", "Where to save the graph definition.")

FLAGS = tf.app.flags.FLAGS

# Game parameters.
NUM_PIECE_TYPES = 6
MAX_MOVES = 100

# Dimension of the input features.
BOARD_FEATURES_DIM = 41  # Should match ai.AllFeaturesDim

# These should match the same in policy_features.go
ACTION_FEATURES_DIM = 1  # Static/context features.
NUM_SECTIONS = 6  # Sections of neighbourhood.
POSITIONS_PER_SECTION = 3  # Num of board positions per section.
FEATURES_PER_POSITION = 16  # Num of features per position.
# NEIGHBOURHOOD_NUM_FEATURES = 305
NEIGHBOURHOOD_NUM_FEATURES = (
		(1 + POSITIONS_PER_SECTION * NUM_SECTIONS) * FEATURES_PER_POSITION + ACTION_FEATURES_DIM)

# Neural Network parameters for board embedding and value prediction.
BOARD_NUM_HIDDEN_LAYERS = 2
BOARD_NODES_PER_LAYER = 128 - BOARD_FEATURES_DIM
BOARD_EMBEDDING_DIM = 64 - BOARD_FEATURES_DIM

# FFNN parameters for actions classifier.
ACTIONS_NUM_HIDDEN_LAYERS = 4
ACTIONS_NODES_PER_LAYER = 128

# Full board convolutions
FULL_BOARD_CONV_DEPTH = 64
FULL_BOARD_CONV_LAYERS = 8

# Neural Network parameters for actions embedding and value prediction.
NEIGHBOURHOOD_NUM_HIDDEN_LAYERS = 3
NEIGHBOURHOOD_NODES_PER_LAYER = 384 - NEIGHBOURHOOD_NUM_FEATURES
NEIGHBOURHOOD_EMBEDDING_DIM = 32

# ACTIVATION=tf.nn.selu
ACTIVATION = tf.nn.leaky_relu

NORMALIZATION = None
NORMALIZATION = tf.layers.batch_normalization


def BuildBoardEmbeddings(board_features, initializer, l2_regularizer):
	global ACTIVATION
	with tf.name_scope("BuildBoardEmbeddings"):
		with tf.variable_scope("board_kernel", reuse=tf.AUTO_REUSE):
			logits = hive_lib.build_skip_ffnn(board_features, BOARD_NUM_HIDDEN_LAYERS, BOARD_NODES_PER_LAYER,
											  True, BOARD_EMBEDDING_DIM, ACTIVATION, initializer, l2_regularizer)
	return logits


def BuildFullBoardConvolutions(full_board):
	global ACTIVATION
	with tf.name_scope("BuildFullBoardConvolutions"):
		for ii in range(FULL_BOARD_CONV_LAYERS):
			with tf.variable_scope("hex_layer_{}".format(ii), reuse=tf.AUTO_REUSE):
				full_board = hive_lib.hexagonal_layer(full_board, ACTIVATION, FULL_BOARD_CONV_DEPTH)
		full_board_max = tf.math.reduce_max(full_board, axis=[1, 2], name="full_board_max")
		full_board_mean = tf.math.reduce_mean(full_board, axis=[1, 2], name="full_board_mean")
		full_board_sum = tf.math.reduce_sum(full_board, axis=[1, 2], name="full_board_sum")
		return (full_board,
				tf.concat([full_board_max, full_board_mean, full_board_sum], axis=1))


def UnsupervisedBoardLoss(full_board_embedding, board_features, initializer, l2_regularizer):
	"""Create a loss to the ability of full_board_embedding to predict board_features."""
	global ACTIVATION
	with tf.name_scope("UnsupervisedBoardLoss"):
		with tf.variable_scope("unsupervised_board_kernel"):
			width_full_board = full_board_embedding.shape[1]
			embedding = hive_lib.build_skip_ffnn(full_board_embedding, 1, 128,
												 False, BOARD_FEATURES_DIM, ACTIVATION, initializer, l2_regularizer)
		with tf.variable_scope("unsupervised_board_output"):
			predictions = tf.layers.dense(embedding, BOARD_FEATURES_DIM, activation=None,
										  name="linear_layer", kernel_initializer=initializer,
										  kernel_regularizer=l2_regularizer)
		unsupervised_loss = tf.losses.mean_squared_error(labels=board_features, predictions=predictions,
														 reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
		return unsupervised_loss


def BuildBoardModel(board_embeddings, board_labels, initializer, l2_regularizer):
	with tf.name_scope("BuildBoardModel"):
		with tf.variable_scope("board_output"):
			board_values = tf.layers.dense(board_embeddings, 1, activation=None,
										   name="linear_layer", kernel_initializer=initializer,
										   kernel_regularizer=l2_regularizer)
		# Adjust prediction.
		board_predictions = hive_lib.sigmoid_to_max(board_values)
		board_labels = tf.cast(board_labels, MODEL_DTYPE)
		reshaped_labels = tf.reshape(board_labels, [-1, 1])
		board_losses = tf.losses.mean_squared_error(reshaped_labels, board_values,
													reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
		return (board_predictions, board_losses)


def BuildNeighbourhoodEmbeddings(actions_features, center, neighbourhood, initializer, l2_regularizer):
	global ACTIVATION
	with tf.name_scope("BuildNeighbourhoodEmbeddings"):
		rotation_embeddings = []
		with tf.variable_scope("neighbourhood_kernel", reuse=tf.AUTO_REUSE):
			for rotation in range(6):
				neigh_rotated = neighbourhood
				if rotation > 1:
					neigh_rotated = tf.manip.roll(
						neighbourhood, shift=rotation, axis=1)
				neigh_concated = tf.reshape(neigh_rotated, [
					tf.shape(neigh_rotated)[0], neighbourhood.shape[1] * neighbourhood.shape[2]])
				all = tf.concat(
					[actions_features, center, neigh_concated], axis=1)
				embedding = hive_lib.build_skip_ffnn(all, NEIGHBOURHOOD_NUM_HIDDEN_LAYERS,
													 NEIGHBOURHOOD_NODES_PER_LAYER,
													 False, NEIGHBOURHOOD_EMBEDDING_DIM, ACTIVATION,
													 initializer, l2_regularizer)
				rotation_embeddings.append(embedding)
		all_embeddings = tf.stack(rotation_embeddings, axis=1)
		sum = tf.reduce_sum(all_embeddings, axis=1)
		max = tf.reduce_max(all_embeddings, axis=1)
		return tf.concat([sum, max], axis=1)


def DereferencePositionEmbedding(full_board_embeddings, actions_board_indices, positions, name):
	"""Gather embeddings from board/positions given."""
	actions_board_indices = tf.expand_dims(actions_board_indices, -1)
	indices = tf.concat([actions_board_indices, positions], axis=1)
	return tf.manip.gather_nd(full_board_embeddings, indices, name)


def BuildActionsModel(
		full_board_embeddings, all_board_embeddings,
		actions_board_indices, actions_is_move,
		actions_src_positions,
		actions_tgt_positions,
		actions_pieces, 
		actions_labels, initializer, l2_regularizer):
	with tf.name_scope('BuildActionsModel'):
		# De-reference (gather) embeddings for locations where from/to 
		# action is happening.
		src_embeddings = DereferencePositionEmbedding(
			full_board_embeddings, actions_board_indices, actions_src_positions,
			name='actions_src_embeddings_gather_nd')
		src_embeddings = tf.where(
			actions_is_move,
			src_embeddings,
			tf.zeros_like(src_embeddings),
			name='actions_src_embedding_filtering')
		tgt_embeddings = DereferencePositionEmbedding(
			full_board_embeddings, actions_board_indices, actions_tgt_positions,
			name='actions_tgt_embeddings_gather_nd')

		# Concat all relevant action features.
		actions_is_move_feature = tf.expand_dims(tf.cast(actions_is_move, dtype=MODEL_DTYPE), -1)
		gathered_board_embeddings = tf.gather(
			all_board_embeddings, actions_board_indices, name='dereference_all_board_embeddings_to_actions')
		actions_pieces = tf.cast(actions_pieces, dtype=MODEL_DTYPE) 
		actions_all_features = tf.concat([
			gathered_board_embeddings,
			actions_is_move_feature, src_embeddings, tgt_embeddings, actions_pieces], 
			axis=1, name='actions_all_features_concat')

		# Build loss and predictions.
		actions_labels = tf.cast(actions_labels, MODEL_DTYPE)
		with tf.variable_scope("actions_kernel"):
			embeddings = hive_lib.build_skip_ffnn(
				actions_all_features, 
				ACTIONS_NUM_HIDDEN_LAYERS, ACTIONS_NODES_PER_LAYER,
				False, ACTIONS_NODES_PER_LAYER, ACTIVATION, initializer, l2_regularizer)
			actions_logits = tf.layers.dense(inputs=embeddings, units=1, activation=None, name="final_linear_layer",
											 kernel_initializer=initializer, kernel_regularizer=l2_regularizer)
		log_soft_max = hive_lib.sparse_log_soft_max(tf.reshape(actions_logits, [-1]), actions_board_indices)
		actions_predictions = tf.exp(log_soft_max)
		actions_loss = tf.reduce_sum(hive_lib.sparse_cross_entropy_loss(log_soft_max, actions_labels))
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
	initializer = tf.contrib.layers.variance_scaling_initializer(
		factor=1.0, mode='FAN_IN')
	initializer = None

	# Build board inputs
	is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
	learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
	# unsupervised_loss_ratio = tf.placeholder(tf.float32, shape=(), name='self_supervision')
	l2_regularization = tf.placeholder(
		tf.float32, shape=(), name='l2_regularization')
	l2_regularizer = BuildRegularizer(l2_regularization)

	# Board data.
	board_features = tf.placeholder(
		tf.float32, shape=[None, BOARD_FEATURES_DIM], name='board_features')
	full_board = tf.placeholder(
		tf.float32, shape=[None, None, None, FEATURES_PER_POSITION],
		name="full_board",
	)
	board_labels = tf.placeholder(
		tf.float32, shape=[None], name='board_labels')
	hive_lib.report_tensors('Board inputs:', [
		is_training, learning_rate, board_features, full_board, board_labels
	])

	# Add Batch Normalization to the activation function.
	global NORMALIZATION
	if NORMALIZATION is not None:
		global ACTIVATION
		prev_activation = ACTIVATION

		def normalized_activation(x):
			with tf.variable_scope("batch_norm"):
				return prev_activation(
					# tf.layers.batch_normalization(
					tf.contrib.layers.batch_norm(
						inputs=x, is_training=is_training, fused=True
					))

		ACTIVATION = normalized_activation

	# Build board logits and model.
	board_features = tf.cast(board_features, MODEL_DTYPE)
	full_board = tf.cast(full_board, MODEL_DTYPE)
	full_board_embeddings, full_board_pooled_embeddings = BuildFullBoardConvolutions(full_board)
	# unsupervised_board_loss = UnsupervisedBoardLoss(full_board_pooled_embeddings, board_features,
	#                                                 initializer, l2_regularizer)

	board_embeddings = BuildBoardEmbeddings(
		board_features, initializer, l2_regularizer)
	all_board_embeddings = tf.concat([full_board_pooled_embeddings, board_embeddings], axis=1)
	board_predictions, board_losses = BuildBoardModel(
		all_board_embeddings, board_labels, initializer, l2_regularizer)

	# total_losses = board_losses + unsupervised_loss_ratio * unsupervised_board_loss
	total_losses = board_losses 
	board_predictions = tf.identity(
		tf.cast(tf.reshape(board_predictions, [-1]), tf.float32), name='board_predictions')
	board_losses = tf.identity(
		tf.cast(board_losses, tf.float32), name='board_losses')
	hive_lib.report_tensors('Board outputs:', [
		board_predictions, board_losses
	])

	# Build per action inputs.
	# All inputs are sparse, since the number of actions is variable.
	# The input `actions_board_indices` list for each of the other
	# actions_* tensors what is the corresponding board -- so the indices
	# are from 0 to len(board_features)-1.
	actions_board_indices = tf.placeholder(
		tf.int64, shape=[None], name='actions_board_indices')
	actions_is_move = tf.placeholder(
		tf.bool, shape=[None], name='actions_is_move')
	actions_src_positions = tf.placeholder(
		tf.int64, shape=[None, 2], name='actions_src_positions')
	actions_tgt_positions = tf.placeholder(
		tf.int64, shape=[None, 2], name='actions_tgt_positions')
	actions_pieces = tf.placeholder(
		tf.float32, shape=[None, NUM_PIECE_TYPES], name='actions_pieces')
	actions_labels = tf.placeholder(
		tf.float32, shape=[None], name='actions_labels')

	hive_lib.report_tensors('Actions inputs:', [
		actions_board_indices, actions_is_move, actions_src_positions,
		actions_tgt_positions, actions_pieces, actions_labels
	])

	# Build actions model.
	actions_predictions, actions_losses = BuildActionsModel(
		full_board_embeddings, all_board_embeddings,
		actions_board_indices, 
		actions_is_move, actions_src_positions, actions_tgt_positions, actions_pieces, 
		actions_labels, initializer, l2_regularizer)
	actions_predictions = tf.identity(
		tf.cast(tf.reshape(actions_predictions, [-1]), tf.float32), name='actions_predictions')
	actions_losses = tf.identity(
		tf.cast(actions_losses, tf.float32), name='actions_losses')
	total_losses += actions_losses
	hive_lib.report_tensors('Actions outputs:', [
		actions_predictions, actions_losses
	])

	# Build optimizer and train opt.
	global_step = tf.train.create_global_step()
	# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	with tf.name_scope("capped_grad_loss_wrt_vars"):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		grads_and_vars = optimizer.compute_gradients(total_losses)
		capped_grads_and_vars = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grads_and_vars]

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.apply_gradients(capped_grads_and_vars, name='train')

	init = tf.global_variables_initializer()

	print('Training:')
	print('\tInitialize variables:\t', init.name)
	print('\tTrain one step:\t', train_op.name)

	# Mean loss: more stable across batches of different sizes.
	mean_loss = total_losses / tf.cast(tf.shape(board_features)[0], dtype=tf.float32)
	mean_loss = tf.identity(tf.cast(mean_loss, tf.float32), name='mean_loss')
	print('\tMean total loss:\t', mean_loss.name)

	# Create saver nodes.
	CreateSaveDef()

	# Save model
	SaveGraph(FLAGS.output)
	print('Saved to {}'.format(FLAGS.output))


if __name__ == '__main__':
	tf.app.run()
