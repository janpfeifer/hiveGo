#!/usr/bin/python3
# This will build an empty base model, with training ops that can be executed from Go.
import tensorflow as tf

# Dimension of the input features, should match ai.AllFeaturesDim
ALL_FEATURES_DIM = 37
NUM_LAYERS = 4
NODES_PER_LAYER = 128 - ALL_FEATURES_DIM
LEARNING_RATE = 0.01

# Batch of input and target output (1x1 matrices)
x = tf.placeholder(tf.float32, shape=[None, ALL_FEATURES_DIM], name='input')
label = tf.placeholder(tf.float32, shape=[None], name='label')
learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

# Feed Forward Neural Network, with connections to the input at every layer.
logits = tf.layers.dense(x, NODES_PER_LAYER, tf.nn.selu)
for _ in range(NUM_LAYERS - 1):
    logits = tf.concat([logits, x], 1)
    logits = tf.layers.dense(logits, NODES_PER_LAYER, tf.nn.selu)
y_ = tf.layers.dense(logits, 1)
predict = tf.reshape(y_, [-1], name='output')

# Optimize loss
loss = tf.losses.mean_squared_error(tf.reshape(label, [-1, 1]), y_)
loss = tf.identity(loss, name='loss')
global_step = tf.train.create_global_step()
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step, name='train')


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
