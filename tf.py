# -*- coding: utf-8 -*-

# # basics

# $$a=(b+c)*(c+2)$$
#
# ![title](graph.png)
#
# * http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
# * http://adventuresinmachinelearning.com/neural-networks-tutorial/
# * https://beckernick.github.io/neural-network-scratch/
#

# In[1]:

import numpy as np

import tensorflow as tf

#
# # first, create a TensorFlow constant
# const = tf.constant(2.0, name="const")
#
# # create TensorFlow variables
# #b = tf.Variable(2.0, name='b')
# b = tf.placeholder(tf.float32, [None, 1], name='b')
# c = tf.Variable(1.0, name='c')
#
#
# # In[2]:
#
# # now create some operations
# d = tf.add(b, c, name='d')
# e = tf.add(c, const, name='e')
# a = tf.multiply(d, e, name='a')
# # create TensorFlow variables
#
#
# # In[3]:
#
# # setup the variable initialisation
# init_op = tf.global_variables_initializer()
#
#
# # In[4]:
#
#
# sess = tf.Session()
#
# sess.run(init_op)
# '''d__ = sess.run(d)
# e__ = sess.run(e)
# a__= sess.run(a)'''
# a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
# #print("a:{0}\td:{1}\te:{2}\ta_out:{3}".format(a__,d__,e__,a_out))
# print(a_out)
#
#
# # In[29]:
#
# np.arange(0, 10)[:, np.newaxis]
#

# In[109]:

#es = np.ceil(2 * np.random.random((10,15)) +5)



# In[110]:





# https://stackoverflow.com/questions/41704484/what-is-difference-between-tf-truncated-normal-and-tf-random-normal#43726140

# In[80]:

# Neural network variables

learning_rate = 0.1
epochs = 10
batch_size = 100

NODES_INPUT = 10
NODES_H1 = 15
NODES_H2 = 7
NODES_OUPUT = 10
NUM_OBS = 40


es = np.random.uniform(0,15,size=[NUM_OBS,NODES_INPUT]).astype(np.float32)
print(es)
#na = 2 * np.random.random((10,10)) - 1
na = np.ceil(np.random.uniform(0,3,size=[NUM_OBS,1])).astype(np.float32)
print(na)

# In[81]:

# entrada: X -> español
# salida: y -> náhuatl

X = tf.placeholder(shape=[None,NODES_INPUT], dtype=tf.float32,name='X')
y = tf.placeholder(shape=[None,NODES_OUPUT], dtype=tf.float32,name='y')

# In[82]:

# capa oculta input to the hidden layer
layer1 = {"W1":tf.Variable(tf.random_normal([NODES_INPUT,NODES_H1],stddev=0.01),name = 'W1'),
          "b1":tf.Variable(tf.random_normal([NODES_H1]), name = 'b1')}
# W1 = tf.Variable(tf.random_normal([NODES_INPUT,NODES_H1],stddev=0.01),name = 'W1')
# b1 = tf.Variable(tf.random_normal([NODES_H1]), name = 'b1')

# capa de salida hidden layer to the output layer
output_layer = {"W2":tf.Variable(tf.random_normal([NODES_H1,NODES_OUPUT],stddev=0.01),name = 'W2'),
                "b2":tf.Variable(tf.random_normal([NODES_OUPUT]), name = 'b2')}
# W2 = tf.Variable(tf.random_normal([NODES_H1,NODES_OUPUT],stddev=0.01),name = 'W2')
# b2 = tf.Variable(tf.random_normal([NODES_OUPUT]), name = 'b2')


# In[83]:

# calcular salida de capa oculta
# h(x) = x*W + bias
hidden_output = tf.add(tf.matmul(es, layer1["W1"]), layer1["b1"])
# función de activación ReLU
hidden_output = tf.nn.relu(hidden_output)


# In[84]:

# calcular salida la NN
output = tf.add(tf.matmul(hidden_output, output_layer["W2"]),output_layer["b2"])
na_predicted = tf.nn.softmax(output)


# In[90]:

# costo
loss = tf.reduce_mean(tf.squared_difference(na_predicted,y))


# In[91]:

# optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# In[93]:

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(na, 1), tf.argmax(na_predicted, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

