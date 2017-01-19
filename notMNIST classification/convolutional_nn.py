from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def reformat(dataset, labels, num_labels, image_size, num_channels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return(100.0*np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def model(data, layers):
  conv = tf.nn.conv2d(data, layers[0][0], [1, 1, 1, 1], padding="SAME")
  hidden = tf.nn.relu(conv + layers[0][1])
  pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

  conv = tf.nn.conv2d(pool, layers[2][0], [1, 1, 1, 1], padding="SAME")
  hidden = tf.nn.relu(conv + layers[2][1])
  pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

  shape = pool.get_shape().as_list()
  reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
  hidden = tf.nn.relu(tf.matmul(reshape, layers[4][0]) + layers[4][1])

  return tf.matmul(hidden, layers[5][0]) + layers[5][1]

def train_cnn_using_sgd(save, 
  num_steps, 
  batch_size=16, 
  num_hidden=64, 
  dropout_prob=0.95, 
  num_channels=1, 
  patch_size=5, 
  depth = 16):
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  print('Training   set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

  image_size = train_dataset.shape[1]
  num_labels = train_dataset.shape[1]

  train_dataset, train_labels = reformat(train_dataset, train_labels, num_labels, image_size, num_channels)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, num_labels, image_size, num_channels)
  test_dataset, test_labels = reformat(test_dataset, test_labels, num_labels, image_size, num_channels)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    layers = [[None, None] for i in range(6)] 

    layers[0][0] = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layers[0][1] = tf.Variable(tf.zeros([depth]))
    layers[1][0] = tf.Variable(tf.truncated_normal([2, 2, depth, depth], stddev=0.1))
    layers[2][0] = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layers[2][1] = tf.Variable(tf.constant(1.0, shape=[depth]))
    layers[3][0] = tf.Variable(tf.truncated_normal([2, 2, num_channels, depth], stddev=0.1))
    layers[4][0] = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev = 0.1))
    layers[4][1] = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layers[5][0] = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layers[5][1] = tf.Variable(tf.constant(1.0, shape= [num_labels]))
    bias = tf.Variable(tf.zeros([num_labels]))

    logits = model(tf_train_dataset, layers)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(test_dataset, layers))
    valid_prediction = tf.nn.softmax(model(valid_dataset, layers))

  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(num_steps):
      offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
      batch_data = train_dataset[offset:offset+batch_size, :]
      batch_labels = train_labels[offset:offset+batch_size, :]

      tf_feed = {tf_train_dataset:batch_data, tf_train_labels:batch_labels}

      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=tf_feed)
      if (step%50 == 0):
        print("Step %d: loss = %f" % (step, l))
        print("Train accuracy: %.1f%%"%accuracy(predictions, batch_labels))
        print("Valid accuracy: %.1f%%"%accuracy(valid_prediction.eval(), valid_labels))

    print('===> Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))  

if __name__ == "__main__":
  pickle_file = 'notMNIST.pickle'

  save = None
  with open(pickle_file, 'rb') as f:
    save = pickle.load(f)

  #train_using_gd(save, 801, 10000)
  #print("Logistic regression with sgd:")
  #train_log_reg_using_stochastic_gd(save, 801, 128)
  print("NN with sgd:")
  train_cnn_using_sgd(save, 1001, batch_size=512)
