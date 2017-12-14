# -*- coding: utf-8 -*-

# # basics

# $$a=(b+c)*(c+2)$$
#
# ![title](graph.png)
#
# * http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
# * http://adventuresinmachinelearning.com/neural-networks-tutorial/
# * https://beckernick.github.io/neural-network-scratch/
# * https://stackoverflow.com/questions/41704484/what-is-difference-between-tf-truncated-normal-and-tf-random-normal#43726140

# %%

__author__ = "Olivares"
import numpy as np
import tensorflow as tf


# %%

# Neural network variables

LEARNING_RATE = 0.1
EPOCHS = 10
BATCH_SIZE = 100

NODES_INPUT = 10
NODES_H1 = 15
NODES_H2 = 7
NODES_OUPUT = 10
NUM_OBS = 40

# %%
# Datos
es = np.random.uniform(0,15,size=[NUM_OBS,NODES_INPUT]).astype(np.float32)
na = np.ceil(np.random.uniform(0,3,size=[NUM_OBS,1])).astype(np.float32)

print("Vector es:")
print(es)
print("Vector na:")
print(na)

# %%
# Placeholders para datos de entrada 
# entrada: X -> español
# salida: y -> náhuatl

X = tf.placeholder(shape=[None,NODES_INPUT], dtype=tf.float32,name='X')
y = tf.placeholder(shape=[None,NODES_OUPUT], dtype=tf.float32,name='y')

# %%

# capa oculta input to the hidden layer
layer1 = {"W1":tf.Variable(tf.random_normal([NODES_INPUT,NODES_H1],stddev=0.01),name = 'W1'),
          "b1":tf.Variable(tf.random_normal([NODES_H1]), name = 'b1')}

output_layer = {"W2":tf.Variable(tf.random_normal([NODES_H1,NODES_OUPUT],stddev=0.01),name = 'W2'),
                "b2":tf.Variable(tf.random_normal([NODES_OUPUT]), name = 'b2')}
# W2 = tf.Variable(tf.random_normal([NODES_H1,NODES_OUPUT],stddev=0.01),name = 'W2')
# b2 = tf.Variable(tf.random_normal([NODES_OUPUT]), name = 'b2')


# %%

# Se calcula la matriz de la capa de entrada a la capa oculta 
# h(x) = W(x) + bias
hidden_output = tf.add(tf.matmul(es, layer1["W1"]), layer1["b1"])
# función de activación ReLU
hidden_output = tf.nn.relu(hidden_output)


# %%

# calcular salida la NN
output = tf.add(tf.matmul(hidden_output, output_layer["W2"]),output_layer["b2"])
na_predicted = tf.nn.softmax(output)


# %%

# costo
loss = tf.reduce_mean(tf.squared_difference(na_predicted,y))


# %%

# optimiser
optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE=LEARNING_RATE).minimize(loss)


# %%

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(na, 1), tf.argmax(na_predicted, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

