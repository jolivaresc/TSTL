# coding: utf-8

# In[ ]:


import tensorflow as tf
import utils
# from math import exp
__author__ = "Olivares Castillo José Luis"

# reset everything to rerun in jupyter
tf.reset_default_graph()


# In[ ]:


print("TensorFlow v{}".format(tf.__version__))


# # Semilla para reproducibilidad

# In[ ]:


tf.set_random_seed(42)


# # Cargar vectores desde archivos.
# Leer archivos node2vec

# In[ ]:


es, na = utils.load_node2vec()
print("es:", es.shape, "\tna:", na.shape)


# Se buscan los índices de los lexicones semilla dentro de los dataframes para poder
# acceder a sus representaciones vectoriales.

# In[ ]:


index_es, index_na = utils.get_seed_index(es, na)
print("index_es:", index_es.__len__(), "index_na:", index_na.__len__())


# Se obtienen los vectores de los lexicones semilla.

# In[ ]:


es_vectores = utils.get_vectors(es, index_es)
na_vectores = utils.get_vectors(na, index_na)


# # Hyperparameters

# In[ ]:


tf.set_random_seed(42)
tf.reset_default_graph()
print(tf.test.gpu_device_name())

LEARNING_RATE = 1
EPOCHS = 1000
# Dimensión de vectores de entrada (número de neuronas en capa de entrada).
NODES_INPUT = es_vectores[0].size

# Número de neuronas en capas ocultas.
NODES_H1 = 350  # 70 - 20 - 15
NODES_H2 = 240  # 42 - 20
NODES_H3 = 220  # 42 - 20
NODES_H4 = 200  # 42 - 20
NODES_OUTPUT = na_vectores[0].size

# Inicializar pesos usando xavier_init
XAVIER_INIT = False

model = "model2143"


with tf.name_scope('input'):
    # El valor None indica que se puede modificar la dimensión de los tensores
    # por si se usan todos los vectores o batches.
    X = tf.placeholder(shape=[None, NODES_INPUT],
                       dtype=tf.float64, name='input_es')
    y = tf.placeholder(shape=[None, NODES_OUTPUT],
                       dtype=tf.float64, name='target_na')


def fully_connected_layer(input, size_in, size_out, name, xavier_init=True, stddev=0.1, dtype=tf.float64):
    with tf.name_scope(name):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        if xavier_init:
            W = tf.get_variable(name="W" + name, shape=[size_in, size_out], dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer(
                                    dtype=dtype),
                                use_resource=True,
                                regularizer=regularizer)
            #W = tf.nn.l2_normalize(W, [0,1])
        else:
            W = tf.Variable(tf.truncated_normal(
                [size_in, size_out], stddev=stddev, dtype=dtype), name="W")
            #W = tf.nn.l2_normalize(W, [0,1])
        # Bias.
        b = tf.Variable(tf.constant(
            0.1, shape=[size_out], dtype=dtype), name="b")

        # h(x) = (input * weights) + bias
        output = tf.nn.xw_plus_b(input, W, b)
        # visualizarlos en TensorBoard.
        tf.summary.histogram("weights", W)
        tf.summary.histogram("pre_activations", output)

        return output


def activation_function(layer, act, name, alpha=tf.constant(0.2, dtype=tf.float64)):
    if act == "leaky_relu":
        return tf.nn.leaky_relu(layer, alpha, name=name)

    elif act == "softmax":
        return tf.nn.softmax(layer, name=name)

    elif act == "sigmoid":
        return tf.nn.sigmoid(layer, name=name)

    elif act == "tanh":
        return tf.nn.tanh(layer, name=name)

    return tf.nn.relu(layer, name=name)


# Se definen las capas.

W1 = tf.get_variable(name="W1", shape=[NODES_INPUT, NODES_H1], dtype=tf.float64,
                     initializer=tf.contrib.layers.xavier_initializer(
                         dtype=tf.float64),
                     use_resource=True,
                     regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
b1 = tf.Variable(tf.constant(
    0.1, shape=[NODES_H1], dtype=tf.float64), name="b1")

fc1 = tf.nn.xw_plus_b(X, W1, b1)
# Se calcula la salida de la capa.
#fc1 = fully_connected_layer(X, NODES_INPUT, NODES_H1, "fc1", xavier_init=XAVIER_INIT)

# Activación de la capa.
fc1 = activation_function(fc1, "relu", "fc1")


# Se añade histograma de activación de la capa para visualizar en
# TensorBoard.
tf.summary.histogram("fc1/relu", fc1)

"""
#2nd layer
fc2 = fully_connected_layer(fc1, NODES_H1, NODES_H2, "fc2", xavier_init=XAVIER_INIT)
fc2 = activation_function(fc2, "relu", "fc2")
tf.summary.histogram("fc2/relu", fc2)


#3rd layer
fc3 = fully_connected_layer(fc2, NODES_H2, NODES_H3, "fc3", xavier_init=XAVIER_INIT)
fc3 = activation_function(fc3, "relu", "fc3")
tf.summary.histogram("fc3/relu", fc3)

fc4 = fully_connected_layer(fc3,NODES_H3,NODES_H4,"fc4",xavier_init=XAVIER_INIT)
fc4 = activation_function(fc4,"relu","fc4")
tf.summary.histogram("fc4/relu", fc4)
"""
W_na = tf.get_variable(name="W_na", shape=[NODES_H1, NODES_OUTPUT], dtype=tf.float64,
                       initializer=tf.contrib.layers.xavier_initializer(
                           dtype=tf.float64),
                       use_resource=True,
                       regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
b_na = tf.Variable(tf.constant(
    0.1, shape=[NODES_OUTPUT], dtype=tf.float64), name="b_na")
#Wp = tf.transpose(W1)

nah_predicted = tf.nn.xw_plus_b(fc1, W_na, b_na)
#nah_predicted = fully_connected_layer(fc1, NODES_H1, NODES_OUTPUT, "output",xavier_init=XAVIER_INIT)
#nah_predicted = fully_connected_layer(fc4, NODES_H4, NODES_OUTPUT, "output", xavier_init=XAVIER_INIT)
#nah_predicted = activation_function(nah_predicted, "sigmoid", "output")
#tf.summary.histogram("output/tanh", nah_predicted)
tf.summary.histogram("output", nah_predicted)


#regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W1)

#http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/

# Loss
loss = tf.reduce_mean(tf.squared_difference(nah_predicted, y), name="loss")

'''# Loss function with L2 Regularization with beta=0.01
regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
loss = tf.reduce_mean(loss + 0.01 * regularizers)'''


tf.summary.scalar("loss", loss)


optimiser = tf.train.MomentumOptimizer(
    learning_rate=LEARNING_RATE, momentum=0.5)
#optimiser = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE)

# Compute gradients
gradients, variables = zip(*optimiser.compute_gradients(loss))

gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

# Apply processed gradients to optimizer.
train_op = optimiser.apply_gradients(zip(gradients, variables))


# Accuracy
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # Se compara salida de la red neuronal con el vector objetivo.
        correct_prediction = tf.equal(
            tf.argmax(nah_predicted, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        # Se calcula la precisión.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    tf.summary.scalar('accuracy', accuracy)


# In[ ]:

LOGPATH = "logs/modeld"
print("logpath:", LOGPATH)


# Se crea la sesión
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Se ponen los histogramas y valores de las gráficas en una sola variable.
summaryMerged = tf.summary.merge_all()

# Escribir a disco el grafo generado y las gráficas para visualizar en TensorBoard.
writer = tf.summary.FileWriter(LOGPATH, sess.graph)

# Se inicializan los valores de los tensores.
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Ejecutando sesión
sess.run(init)


for i in range(EPOCHS):
    _loss, _, sumOut = sess.run([loss, train_op, summaryMerged], feed_dict={
        X: es_vectores, y: na_vectores})

    writer.add_summary(sumOut, i)

    if (i % 700) == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={
                                       X: es_vectores, y: na_vectores})
        print("Epoch:", i, "/", EPOCHS, "\tLoss:",
              _loss, "\tAccuracy:", train_accuracy)


SAVE_PATH = "./" + model + ".ckpt"
print("save path", SAVE_PATH)
#save_model = saver.save(sess, SAVE_PATH)
print("Model saved in file: %s", SAVE_PATH)
writer.close()
