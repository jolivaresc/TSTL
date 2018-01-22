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


LEARNING_RATE = 1.0
# Dimensión de vectores de entrada (número de neuronas en capa de entrada).
NODES_INPUT = es_vectores[0].size

# Número de neuronas en capas ocultas.
NODES_H1 = 90  # 70 - 20 - 15
NODES_H2 = 74  # 42 - 20
NODES_H3 = 74  # 42 - 20
NODES_H4 = 90  # 42 - 20
NODES_OUPUT = na_vectores[0].size

# Inicializar pesos usando xavier_init
XAVIER_INIT = True

EPOCHS = 370000

# Ruta donde se guarda el grafo para visualizar en TensorBoard.


# # Placeholders
#
# Tensores donde estarán los vectores de entrada y salida.
#
# * X: Vectores de español.
# * y: Vectores de náhuatl.
#
# `tf.name_scope` se utiliza para mostrar las entradas del grafo computacional en `TensorBoard`.

# In[ ]:


with tf.name_scope('input'):
    # El valor None indica que se puede modificar la dimensión de los tensores
    # por si se usan todos los vectores o batches.
    X = tf.placeholder(shape=[None, NODES_INPUT],
                       dtype=tf.float64, name='input_es')
    y = tf.placeholder(shape=[None, NODES_OUPUT],
                       dtype=tf.float64, name='target_na')

"""
with tf.name_scope("dropout"):
    # Placeholder para aplicar dropout a las capas de la red.
    pkeep = tf.placeholder(tf.float64)

with tf.name_scope("lr_decay"):
    lr = tf.placeholder(tf.float64)"""


# # Función para crear las capas de la red.
#
#
# Función para crear capas.

# Args:
# * input (Tensor): Tensor de entrada a la capa.
# * size_in, size_out (int): Dimensiones de entrada y salida de la capa.
# * name (str): Nombre de la capa. Default: fc.
# * stddev (float): Desviación estándar con la que se inicializan los pesos de la capa.
# * dtype: Floating-point representation.

# Returns:
# * act (Tensor): $(input * weights) + bias $
#
#
#

# In[ ]:


def fully_connected_layer(input, size_in, size_out, name,
                          xavier_init=True, stddev=0.1, dtype=tf.float64):
    """Función para crear capas.
    
    Arguments:
        input {Tensor} -- Tensor de entrada a la capa.
        size_in {int}, size_out {int} -- Dimensiones de entrada y salida de la capa.
        name {str} -- Nombre de la capa. Default: fc.
    Keyword Arguments:
        xavier_init {bool} -- Inicializar pesos usando el método Xavier.
                              Xavier Glorot and Yoshua Bengio (2010)
        stddev {float} -- Desviación estándar con la que se inicializan los pesos de la capa. (default: {0})
        dtype {function} -- Floating-point representation. (default: {tf.float64})
    
    Returns:
        Tensor -- Salida de la capa: (input * Weights) + bias
    """

    with tf.name_scope(name):
        # Tensor de pesos.
        """This initializer is designed to keep the scale of the gradients 
        roughly the same in all layers. In uniform distribution this ends up 
        being the range: x = sqrt(6. / (in + out)); [-x, x] and for normal 
        distribution a standard deviation of sqrt(2. / (in + out)) is used.
        Xavier Glorot and Yoshua Bengio (2010)
        http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        """

        if xavier_init:
            W = tf.get_variable(name="W" + name, shape=[size_in, size_out], dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer(
                                    dtype=dtype),
                                use_resource=True)
        else:
            W = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=stddev,
                                                dtype=dtype), name="W")
        # Bias.
        b = tf.Variable(tf.constant(
            0.1, shape=[size_out], dtype=dtype), name="b")

        # h(x) = (input * weights) + bias
        output = tf.nn.xw_plus_b(input, W, b)
        # visualizarlos en TensorBoard.
        tf.summary.histogram("weights", W)
        tf.summary.histogram("pre_activations", output)

        return output


# # Activación de capas.
# Función para activar la salida de las capas.
#
# Args:
# * layer (Tensor): Capa que será activada.
# * name (string): Nombre de la capa para mostrar en `TensorBoard`.
# * act (string): Función de activación. Default: [ReLU](https://www.tensorflow.org/api_docs/python/tf/nn/relu).
# También se pueden utilizar [Leaky ReLU](https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu)
# con un parámetro `alpha = 0.2` por defecto y [Softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax)
# para la capa de salida.
#
# Returns:
#     Capa con función de activación aplicada.
#
# **NOTA:**
# >3.4 Why do we use a leaky ReLU and not a ReLU as an activation function?
# We want gradients to flow while we backpropagate through the network.
# We stack many layers in a system in which there are some neurons
# whose value drop to zero or become negative. Using a ReLU as an activation
# function clips the negative values to zero and in the backward pass,
# the gradients do not flow through those neurons where the values become zero.
# Because of this the weights do not get updated, and the network stops learning
# for those values. So using ReLU is not always a good idea. However, we encourage
# you to change the activation function to ReLU and see the difference.
# [See link](https://www.learnopencv.com/understanding-autoencoders-using-tensorflow-python/)

# In[ ]:


def activation_function(layer, act, name, alpha=tf.constant(0.2, dtype=tf.float64)):
    """Esta función aplica la activación a la capa neuronal.
    Arguments:
        layer {Tensor} -- Capa a activar.
        act {tf.function} -- Función de activación (default: {tf.nn.relu}).
        name {str} -- Nombre para visualización de activación en TensorBoard.
    Keyword Arguments:
        alpha {tf.constant} -- Constante que se usa como argumento para 
                               leaky_relu (default: {tf.constant(0.2)})
        dtype {tf.function} -- Floating-point representation. (default: {tf.float64})
    Returns:
        Tensor -- Capa con función de activación aplicada.
    """

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

# In[ ]:


# Se calcula la salida de la capa.
fc1 = fully_connected_layer(X, NODES_INPUT, NODES_H1, "fc1",
                            xavier_init=XAVIER_INIT)

# Activación de la capa.
fc1 = activation_function(fc1, "sigmoid", "fc1")


# Se añade histograma de activación de la capa para visualizar en
# TensorBoard.
tf.summary.histogram("fc1/sigmoid", fc1)

# Aplicar dropout a la capa para prevenir sobreajustes.
#fc1 = tf.nn.dropout(fc1, pkeep)


# In[ ]:


'''
fc2 = fully_connected_layer(fc1, NODES_H1, NODES_H2,
                            "fc2", xavier_init=XAVIER_INIT)
fc2 = activation_function(fc2, "relu", "fc2")
tf.summary.histogram("fc2/relu", fc2)
#fc2 = tf.nn.dropout(fc2, pkeep)
# In[ ]:
#fc2 = tf.nn.dropout(fc2, pkeep)
fc3 = fully_connected_layer(fc2, NODES_H2, NODES_H3,
                            "fc3", xavier_init=XAVIER_INIT)
fc3 = activation_function(fc3, "relu", "fc3")
tf.summary.histogram("fc3/relu", fc3)


fc4 = fully_connected_layer(fc3, NODES_H3, NODES_H4,
                            "fc4", xavier_init=XAVIER_INIT)
fc4 = activation_function(fc4, "relu", "fc4")
tf.summary.histogram("fc4/relu", fc4)
'''

# In[ ]:


nah_predicted = fully_connected_layer(fc1, NODES_H1, NODES_OUPUT, "output",
                                      xavier_init=XAVIER_INIT)
#nah_predicted = activation_function(output, "sigmoid", "output")
tf.summary.histogram("output/noact", nah_predicted)


# # Función de error
# Se utiliza la función de error por mínimos cuadrados.

# In[ ]:


#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=nah_predicted))
#loss = tf.reduce_mean(tf.reduce_sum((nah_predicted - y) ** 2))
#with tf.name_scope("MSE"):
loss = tf.reduce_mean(tf.squared_difference(nah_predicted, y), name="loss")
tf.summary.scalar("loss", loss)


# # Optimiser
#
# **NOTAS**
# > a) En pruebas, al parecer se presenta el problema de [Vanishing Gradient
# Problem(https://medium.com/@anishsingh20/the-vanishing-gradient-problem-48ae7f501257), la función de
# error parecía quedarse estancada en un mínimo local. Para contrarrestar esto, se utiliza la función
# `tf.clip_by_global_norm` que ajusta el gradiente a un valor específico y evitar que rebase un
# determinado umbral o se haga cero.
# [Ver liga](https://www.tensorflow.org/versions/r0.12/api_docs/python/train/gradient_clipping)
#
# > b) En pruebas, el optimizador para el algoritmo de backpropagation
# [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) se queda
# estancado apenas empieza el entrenamiento (100000 epochs).

# In[ ]:


#https://stackoverflow.com/questions/36498127/how-to-effectively-apply-gradient-clipping-in-tensor-flow

# Create an optimizer.
# TODO: cambiar por AdamOptimizer... upd: Adam se estanca...
#optimiser = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(loss,name="optimizer")
optimiser = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE,momentum=0.5)

# Compute gradients
gradients, variables = zip(*optimiser.compute_gradients(loss))

# For those who would like to understand the idea of gradient clipping(by norm):
# Whenever the gradient norm is greater than a particular threshold,
# we clip the gradient norm so that it stays within the threshold.
# This threshold is sometimes set to 5.
# https://stackoverflow.com/questions/36498127/how-to-effectively-apply-gradient-clipping-in-tensor-flow
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

# Apply processed gradients to optimizer.
train_op = optimiser.apply_gradients(zip(gradients, variables))


# # Accuracy
# Se calcula la precisión de la red neuronal.
#
# - [x] Evaluar con lexicon semilla. (para pruebas de visualización de precisión en `TensorBoard`)
# - [ ] Evaluar con lexicon de evaluación.
#
#
#

# In[ ]:


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

LOGPATH = utils.make_hparam_string(
    "80ACC_Adagrad", "H", NODES_H1, "LR", LEARNING_RATE)
print("logpath:", LOGPATH)


# # TensorFlow Session
#
# Para poder realizar el entrenamiento se debe iniciar una sesión para que se puedan ejecutar las
# operaciones para entrenar y evaluar la red neuronal.

# In[ ]:


# Configuración para pasar como argumento a la sesión de TensorFlow.
# Es para poder ejecutar el grafo en múltiples hilos.
config = tf.ConfigProto(intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=1,
                        #log_device_placement=True
                        )

# Se crea la sesión
sess = tf.Session(config=config)

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


# In[ ]:


for i in range(EPOCHS):
    # learning rate decay
    # https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_2.1_five_layers_relu_lrdecay.py
    """max_learning_rate = 0.45
    min_learning_rate = 0.03
    decay_speed = 5000
    learning_rate = min_learning_rate + \
        # (max_learning_rate - min_learning_rate) * exp(-i / decay_speed)"""

    # Se corre la sesión y se pasan como argumentos la función de error (loss),
    # el optimizador de backpropagation (train_op) y los histogramas (summaryMerged)

    _loss, _, sumOut = sess.run([loss, train_op, summaryMerged],
                                feed_dict={X: es_vectores,
                                           y: na_vectores})
    # Actualiza los histogramas.
    writer.add_summary(sumOut, i)

    # Muestra el valor del error cada 500 pasos de entrenamiento.
    if (i % 500) == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={X: es_vectores,
                                                                y: na_vectores})
        print("Epoch:", i, "/", EPOCHS, "\tLoss:",
              _loss, "\tAccuracy:", train_accuracy)
SAVE_PATH = "./models/model2/Adagrad_H_90_LR_1.0.ckpt"
save_model = saver.save(sess, SAVE_PATH)
print("Model saved in file: %s", SAVE_PATH)
#print("\nAccuracy:", accuracy.eval(feed_dict=feed_dict(es_vectores, na_vectores)))
writer.close()
