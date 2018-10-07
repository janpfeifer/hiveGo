#!/usr/bin/python3
# This will build an empty base model, with training ops that can be executed from Go.
import tensorflow as tf

# Model internal type: tf.float16 presumably is faster in the RX2080 Ti GPU,
# and not slower in others.
MODEL_DTYPE=tf.float16

# Dimension of the input features, should match ai.AllFeaturesDim
ALL_FEATURES_DIM = 41  # Was 39, 37
NUM_LAYERS = 4
NODES_PER_LAYER = 128 - ALL_FEATURES_DIM
LEARNING_RATE = 0.01

def SigmoidTo10(x, smoothness=4.0):
	"""Make a sigmoid curve on values > 9.8 or < -9.8."""
	abs_x = tf.abs(x)
	threshold = tf.constant(9.8, dtype=MODEL_DTYPE)
	mask = (abs_x > threshold)
	sigmoid = tf.sigmoid((abs_x - threshold) / smoothness)
	sigmoid = threshold + (sigmoid - 0.5) * 0.4   # 0.4 = 0.2 / 0.5
	sigmoid = tf.sign(x) * sigmoid
	return tf.where(abs_x > threshold, sigmoid, x)

# Batch of input and target output (1x1 matrices)
x = tf.placeholder(tf.float32, shape=[None, ALL_FEATURES_DIM], name='input')
label = tf.placeholder(tf.float32, shape=[None], name='label')
learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

# Feed Forward Neural Network, with connections to the input at every layer.
x_ = tf.cast(x, MODEL_DTYPE)
logits = tf.layers.dense(x_, NODES_PER_LAYER, tf.nn.selu)
for _ in range(NUM_LAYERS - 1):
    logits = tf.concat([logits, x_], 1)
    logits = tf.layers.dense(logits, NODES_PER_LAYER, tf.nn.selu)
y_ = tf.layers.dense(logits, 1)

# Adjust prediction. 
predict = tf.cast(SigmoidTo10(y_), tf.float32)
predict = tf.reshape(predict, [-1], name='output')

# Optimize loss
label_ = tf.cast(label, MODEL_DTYPE)
loss_ = tf.losses.mean_squared_error(tf.reshape(label_, [-1, 1]), y_)
loss = tf.identity(tf.cast(loss_, tf.float32), name='loss')
global_step = tf.train.create_global_step()
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_, global_step=global_step, name='train')


init = tf.global_variables_initializer()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
saver_def = tf.train.Saver().as_saver_def()

print('Run this operation to initialize variables     : ', init.name)
print('Run this operation for a train step            : ', train_op.name)
print('Feed this tensor to set the checkpoint filename: ',
      saver_def.filename_tensor_name)
print('Run this operation to save a checkpoint        : ',
      saver_def.save_tensor_name)
print('Run this operation to restore a checkpoint     : ',
      saver_def.restore_op_name)
print('Loss tensor name: ', loss.name)
print('Inputs:')
print('\t{}\n\t{}\n\t{}\n'.format(x.name, label.name, learning_rate.name))
# Write the graph out to a file.
with open('tf_model.pb', 'wb') as f:
    f.write(tf.get_default_graph().as_graph_def().SerializeToString())
