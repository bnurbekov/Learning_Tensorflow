from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def reformat(dataset, labels, num_labels, image_size):
  dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return(100.0*np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def train_log_reg_using_gd(save, num_steps, train_subset):
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

  train_dataset, train_labels = reformat(train_dataset, train_labels, num_labels, image_size)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, num_labels, image_size)
  test_dataset, test_labels = reformat(test_dataset, test_labels, num_labels, image_size)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :]) 
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
    bias = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset, weights) + bias
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + bias)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + bias)

  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(num_steps):
      _, l, predictions = session.run([optimizer, loss, train_prediction])
      if (step%100 == 0):
        print("Step %d: loss = %f" % (step, l))
        print("Train accuracy: %.1f%%"%accuracy(predictions, train_labels[:train_subset, :]))
        print("Valid accuracy: %.1f%%"%accuracy(valid_prediction.eval(), valid_labels))

    print('===> Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

def train_log_reg_using_stochastic_gd(save, num_steps, batch_size=128):
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

  train_dataset, train_labels = reformat(train_dataset, train_labels, num_labels, image_size)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, num_labels, image_size)
  test_dataset, test_labels = reformat(test_dataset, test_labels, num_labels, image_size)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
    bias = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset, weights) + bias
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + bias)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + bias)

  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(num_steps):
      offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
      batch_data = train_dataset[offset:offset+batch_size, :]
      batch_labels = train_labels[offset:offset+batch_size, :]

      tf_feed = {tf_train_dataset:batch_data, tf_train_labels:batch_labels}

      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=tf_feed)
      if (step%100 == 0):
        print("Step %d: loss = %f" % (step, l))
        print("Train accuracy: %.1f%%"%accuracy(predictions, batch_labels))
        print("Valid accuracy: %.1f%%"%accuracy(valid_prediction.eval(), valid_labels))

    print('===> Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

def train_nn_using_sgd(save, num_steps, regularization_constant=0.01, batch_size=128, hidden_units_num=1024, dropout_prob=0.95):
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

  train_dataset, train_labels = reformat(train_dataset, train_labels, num_labels, image_size)
  valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, num_labels, image_size)
  test_dataset, test_labels = reformat(test_dataset, test_labels, num_labels, image_size)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

  graph = tf.Graph()
  with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    hidden_weights = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_units_num]))
    hidden_bias = tf.Variable(tf.zeros([hidden_units_num]))
    weights = tf.Variable(tf.truncated_normal([hidden_units_num, num_labels]))
    bias = tf.Variable(tf.zeros([num_labels]))

    hidden_logit = tf.nn.dropout(tf.matmul(tf_train_dataset, hidden_weights) + hidden_bias, dropout_prob)
    logits = tf.matmul(tf.nn.relu(hidden_logit), weights) + bias
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + regularization_constant * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(hidden_weights))

    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, num_steps, 0.95)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_bias), weights) + bias)
    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_bias), weights) + bias)

  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(num_steps):
      offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
      batch_data = train_dataset[offset:offset+batch_size, :]
      batch_labels = train_labels[offset:offset+batch_size, :]

      tf_feed = {tf_train_dataset:batch_data, tf_train_labels:batch_labels}

      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=tf_feed)
      if (step%100 == 0):
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
  train_nn_using_sgd(save, 801, regularization_constant=0.005, batch_size=512)
