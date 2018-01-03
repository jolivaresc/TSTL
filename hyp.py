#http://neuralnetworksanddeeplearning.com/chap3.html

import numpy as np
import tensorflow as tf

# Hyperparameters
LEARNING_RATE = 0.1
EPOCHS = 10
BATCH_SIZE = 100

NODES_INPUT = 50
NODES_H1 = 40
NODES_H2 = 20
NODES_OUPUT = 50
NUM_OBS = 1000

# es - datos de entrada;
# na - datos objetivos
esp = np.random.uniform(0,15,size=[NUM_OBS,NODES_INPUT]).astype(np.float32)
nah = np.random.uniform(0,3,size=[NUM_OBS,NODES_OUPUT]).astype(np.float32)
print(esp.shape,nah.shape)

# entrada: X -> español
# salida: y -> náhuatl
X = tf.placeholder(shape=[None,NODES_INPUT], dtype=tf.float32,name='X')
y = tf.placeholder(shape=[None,NODES_OUPUT], dtype=tf.float32,name='y')
print(X.shape,y.shape)


# Capa oculta (weights & bias)
layer1 = {"W1":tf.Variable(tf.random_normal([NODES_INPUT,NODES_H1],stddev=0.01),name = 'W1'),
          "b1":tf.Variable(tf.random_normal([NODES_H1]), name = 'b1')}

layer2 = {"W2":tf.Variable(tf.random_normal([NODES_H1,NODES_H2],stddev=0.01),name = 'W2'),
          "b2":tf.Variable(tf.random_normal([NODES_H2]), name = 'b2')}

# Capa de salida
output_layer = {"W2":tf.Variable(tf.random_normal([NODES_H2,NODES_OUPUT],stddev=0.01),name = 'W_output'),
                "b2":tf.Variable(tf.random_normal([NODES_OUPUT]), name = 'b_output')}


# Calcular la salida de la 1er capa oculta
# h(x) = x*w + bias
hidden_layer1_output = tf.add(tf.matmul(esp, layer1["W1"]), layer1["b1"],name="hidden1")
# Función de activación usando ReLU
hidden_layer1_output = tf.nn.relu(hidden_layer1_output,name="actfhidden1")

# Calcular la salida de la 2da capa oculta
# h(x) = x*w + bias
hidden_layer2_output = tf.add(tf.matmul(hidden_layer1_output, layer2["W2"]), layer2["b2"],name="hidden2")
# Función de activación usando ReLU
hidden_layer2_output = tf.nn.relu(hidden_layer2_output,name="actfhidden2")


# calcular la salida la NN
output = tf.add(tf.matmul(hidden_layer2_output, output_layer["W2"]),output_layer["b2"],name="output")
# Función de activación usando softmax
nah_predicted = tf.nn.softmax(output,name="actfoutput")

# Función de error (Mean Square Error)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = nah_predicted))
loss = tf.reduce_mean(tf.squared_difference(nah_predicted,y),name="loss_f")
#loss = tf.reduce_sum((nah_predicted - y) ** 2)

# optimiser, 
optimiser = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE,name="GradientDescent").minimize(loss,name="loss")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("./logs/NN",sess.graph)
    for _ in range(5000):
        _, c = sess.run([optimiser,loss],feed_dict={X: esp, y: nah})
        print(c)
    

#tensorboard --logdir=logs