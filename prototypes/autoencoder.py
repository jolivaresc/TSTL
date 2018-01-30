# -*- coding: utf-8 -*-
"""autoencoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebook#fileId=1eEw3YSM-XGLjUhrkrWKYQGxDAXaqlJdj
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from google.colab import files

__author__ = "Olivares Castillo José Luis"

print(tf.__version__, tf.test.gpu_device_name())

##############################
# Cargar archivos necesarios #
##############################


"""'newlexiconb.lst'"""
# Cargar lexicon semilla
# `lexiconfile` es de type dict.
lexiconfile = files.upload()

# Se convierte el archivo a una lista
lexiconsemilla = lexiconfile['newlexiconb.lst'].decode("utf-8").split("\n")

#tuple(lexicon[0].split())


# Separar cada elemento de lista en una tupla
lexicon_ = list()
for i in lexicon:
  lexicon_.append(tuple(i.split()))

# Crear dataframe con los pares traducción
lexicon_df = pd.DataFrame.from_records(lexicon_, columns=["esp", "nah"])
lexicon_df.shape

# Cargar vectores náhuatl
na_n2v = files.upload()

# Cargar vectores español
es_n2v = files.upload()

# Convertir archivos a una lista
es_n2v = es_n2v['es.node2vec.embeddings'].decode("utf-8").split("\n")
na_n2v = na_n2v['na.node2vec.embeddings'].decode("utf-8").split("\n")

# Separar cada elemento de la lista en una tupla
na_tmp = list()
for i in na_n2v:
    if i == 0:
        pass
    else:
        na_tmp.append(tuple(i.split()))

# Eliminar el primer elemento, no se utiliza
na_tmp.pop(0)

# Separar cada elemento de la lista en una tupla
es_tmp = list()
for i in es_n2v:
    if i == 0:
        pass
    else:
        es_tmp.append(tuple(i.split()))

# Eliminar el primer elemento, no se utiliza
es_tmp.pop(0)

# Crear dataframes con los vectores
es_df = pd.DataFrame.from_records(es_tmp)
na_df = pd.DataFrame.from_records(na_tmp)

es_df.head()

na_df.head()

# obtener listas de semillas de español y náhuatl
semillas_esp = list(lexicon_df["esp"].values)
semillas_nah = list(lexicon_df["nah"].values)

semillas_esp.__len__()

# Obtener los índices que tienen las semillas dentro del dataframe de vectores
index_esp = [int(es_df[es_df[0] == palabra].index[0])
             for palabra in semillas_esp]
index_nah = [int(na_df[na_df[0] == palabra].index[0])
             for palabra in semillas_nah]

print(len(index_esp), len(index_nah))


def get_vectors(dataframe, index, format=np.float64):
    """
    Retorna los vectores dentro del dataframe.
    
    Args:
        dataframe (Pandas.dataframe): Contiene las palabras y su representación vectorial.
        index (list): Contiene los índices que se necesitan del dataframe.
        format (numpy format): Tipo flotante. Default float64.
    
    Returns:
        Numpy array: Matriz con representaciones vectoriales.
    """

    return np.array([(dataframe.iloc[_].loc[1::])
                     for _ in index]).astype(format)

# Obtener representaciones vectoriales de las semillas.
es_vectores = get_vectors(es_df, index_esp)
na_vectores = get_vectors(na_df, index_nah)

len(es_vectores)

#################
# Entrenamiento #
#################

tf.set_random_seed(42)
tf.reset_default_graph()
print(tf.test.gpu_device_name())

LEARNING_RATE = 1.0
EPOCHS = 60000
# Dimensión de vectores de entrada (número de neuronas en capa de entrada).
NODES_INPUT = es_vectores[0].size

# Número de neuronas en capas ocultas.
NODES_H1 = 256  # 70 - 20 - 15
NODES_H2 = 256  # 42 - 20
NODES_H3 = 256  # 42 - 20
NODES_H4 = 200  # 42 - 20
NODES_OUPUT = na_vectores[0].size

# Inicializar pesos usando xavier_init
XAVIER_INIT = True

model = "model"




with tf.name_scope('input'):
    # El valor None indica que se puede modificar la dimensión de los tensores
    # por si se usan todos los vectores o batches.
    X = tf.placeholder(shape=[None, NODES_INPUT],dtype=tf.float64, name='input_es')
    y = tf.placeholder(shape=[None, NODES_OUPUT],dtype=tf.float64, name='target_na')


def fully_connected_layer(input, size_in, size_out, name, xavier_init=True, stddev=0.1, dtype=tf.float64):
    with tf.name_scope(name):

        if xavier_init:
            W = tf.get_variable(name="W" + name, shape=[size_in, size_out], dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype), use_resource=True)
        else:
            W = tf.Variable(tf.truncated_normal(
                [size_in, size_out], stddev=stddev, dtype=dtype), name="W")
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


# Se calcula la salida de la capa.
fc1 = fully_connected_layer(X, NODES_INPUT, NODES_H1, "fc1", xavier_init=XAVIER_INIT)

# Activación de la capa.
fc1 = activation_function(fc1, "relu", "fc1")


# Se añade histograma de activación de la capa para visualizar en
# TensorBoard.
tf.summary.histogram("fc1/relu", fc1)


#2nd layer
fc2 = fully_connected_layer(fc1, NODES_H1, NODES_H2, "fc2", xavier_init=XAVIER_INIT)
fc2 = activation_function(fc2, "relu", "fc2")
tf.summary.histogram("fc2/relu", fc2)


#3rd layer
fc3 = fully_connected_layer(fc2, NODES_H2, NODES_H3, "fc3", xavier_init=XAVIER_INIT)
fc3 = activation_function(fc3, "relu", "fc3")
tf.summary.histogram("fc3/relu", fc3)

'''fc4 = fully_connected_layer(fc3,NODES_H3,NODES_H4,"fc4",xavier_init=XAVIER_INIT)
fc4 = activation_function(fc4,"relu","fc4")
tf.summary.histogram("fc4/relu", fc4)'''

#nah_predicted = fully_connected_layer(fc1, NODES_H1, NODES_OUPUT, "output",xavier_init=XAVIER_INIT)
nah_predicted = fully_connected_layer(fc3, NODES_H3, NODES_OUPUT, "output", xavier_init=XAVIER_INIT)
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


optimiser = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.5)

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

LOGPATH = ".logs/model"
print("logpath:", LOGPATH)


# Se crea la sesión
sess = tf.Session()

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
    _loss, _ = sess.run([loss, train_op], feed_dict={
                        X: es_vectores, y: na_vectores})

    #writer.add_summary(sumOut, i)

    if (i % 700) == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={
                                       X: es_vectores, y: na_vectores})
        print("Epoch:", i, "/", EPOCHS, "\tLoss:",
              _loss, "\tAccuracy:", train_accuracy)

SAVE_PATH = "./"+model+".ckpt"
print("save path",SAVE_PATH)
save_model = saver.save(sess, SAVE_PATH)
print("Model saved in file: %s", SAVE_PATH)
writer.close()
#1903
#2034

#############################
# Descargar modelo generado #
#############################

from google.colab import files
files.download("/content/checkpoint")

files.download("/content/"+model+".ckpt.data-00000-of-00001")

files.download("/content/"+model+".ckpt.index")

files.download("/content/"+model+".ckpt.meta")
