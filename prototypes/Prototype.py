
# coding: utf-8

# In[1]:


#http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/deep-neural-nets/
#http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/
#https://jhui.github.io/2017/03/12/TensorBoard-visualize-your-learning/
#https://thecodacus.com/tensorboard-tutorial-visualize-networks-graphically/#.WlVvmnWnG00
#http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6818181
#https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
#https://stackoverflow.com/questions/17241004/pandas-how-to-get-the-data-frame-index-as-an-array
#https://www.tensorflow.org/get_started/graph_viz
#https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
#http://ieeexplore.ieee.org/abstract/document/554199/
#https://stackoverflow.com/questions/36498127/how-to-effectively-apply-gradient-clipping-in-tensor-flow
#https://gertjanvandenburg.com/blog/autoencoder/
#https://jmetzen.github.io/2015-11-27/vae.html
#https://gist.github.com/hussius/1534135a419bb0b957b9
#https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

import tensorflow as tf
from pandas import set_option, read_csv
from numpy import array, float64, set_printoptions
__author__ = "Olivares Castillo José Luis"


# In[2]:


print("TensorFlow v{}".format(tf.__version__))


# In[3]:


# Semilla para reproducibilidad
tf.set_random_seed(42)


# In[4]:


# Se establece la precisión con la que Pandas lee el archivo para evitar que
# trunque el valor de los vectores
set_option('display.max_colwidth', -1)
set_option('precision', 18)


# In[5]:


def load_node2vec():
    """
    Esta función lee los archivos para almacenar los vectores node2vec del español
    y náhuatl los retorna en dataframes de Pandas.
    """

    es = read_csv("../vectors/es.node2vec.embeddings",
                  delimiter=" ", skiprows=1, header=None)
    nah = read_csv("../vectors/na.node2vec.embeddings",
                   delimiter=" ", skiprows=1, header=None)

    return es, nah


# In[6]:


es, nah = load_node2vec()


# In[7]:


es.head()


# In[8]:


nah.head()


# In[9]:


int(es[es[0] == "igual"].index.get_values())


# In[10]:


def get_seed_index(es=es, nah=nah):
    """
    Esta función obtiene los índices de las palabras semillas de los
    dataframes.
    Args:
        es (Dataframe): Contiene vectores n2v de español.
        nah (Dataframe): Contiene vectores n2v de náhuatl.
        
    Returns:
        list (2): Listas con índices de las palabras semillas.
    """

    # Dataframe que contiene las palabras semilla para entrenamiento.
    lexiconsemilla = read_csv("../lexiconessemilla/lexicon.esna.proc.norep.tmp2",
                              delimiter=" ",
                              names=["esp", "nah"])

    # Se almacenan las palabras semillas de español y náhuatl en listas.
    semillas_esp = list(lexiconsemilla["esp"].values)
    semillas_nah = list(lexiconsemilla["nah"].values)

    # Se buscan los índices de las palabras semilla en los dataframes para obtener sus
    # representaciones vectoriales.
    # Nota: Se omite la palabra semilla si no existe su representación vectorial.
    index_esp = [int(es[es[0] == i].index.get_values()) for i in semillas_esp
                 if int(es[es[0] == i].index.get_values().__len__()) == 1]
    index_nah = [int(nah[nah[0] == i].index.get_values()) for i in semillas_nah
                 if int(nah[nah[0] == i].index.get_values().__len__()) == 1]

    return index_esp, index_nah


# In[11]:


index_esp, index_nah = get_seed_index(es, nah)


# In[12]:


print("shape index_esp:", index_esp.__len__(),
      "\nshape index_nah:", index_nah.__len__())


# In[13]:


def get_node2vec(es=es, nah=nah):
    """
    Esta función obtiene los vectores n2v de las palabras semillas, los vectores se castean
    a tipo numpy array.
    
    Args:
        es (Dataframe): Contiene vectores n2v de español.
        nah (Dataframe): Contiene vectores n2v de náhuatl.
        
    Returns:
        numpy array (2): Matriz con los vectores n2v de las palabras semillas.
    """

    es_vectores = array([(es.iloc[index].loc[1::])
                         for index in index_esp]).astype(float64)
    na_vectores = array([(nah.iloc[index].loc[1::])
                         for index in index_nah]).astype(float64)

    return es_vectores, na_vectores


# In[14]:


#es_vectores = array(es[:1900].loc[:,1::]).astype(float64)
#na_vectores = array(nah[:1900].loc[:,1::]).astype(float64)

es_vectores, na_vectores = get_node2vec(es, nah)


# In[15]:


print(es_vectores[0])


# In[16]:


na_vectores


# es_palabras = es[0]
# na_palabras = na[0]

# In[17]:


print("shape es:", es_vectores.shape, "\nshape na:", na_vectores.shape)


# # Hyperparameters

# In[18]:


LEARNING_RATE = 0.7
BATCH_SIZE = 100

NODES_INPUT = es_vectores[0].size
NODES_H1 = 15
NODES_H2 = 15
NODES_OUPUT = na_vectores[0].size
INSTANCES = es_vectores.__len__()
EPOCHS = 100000


# In[19]:


X = tf.placeholder(shape=[None, NODES_INPUT],
                   dtype=tf.float64, name='input_es')
y = tf.placeholder(shape=[None, NODES_OUPUT],
                   dtype=tf.float64, name='target_na')
print("X:", X.shape, "y:", y.shape)


# In[20]:


# Capas ocultas (weights & bias)
hidden_layer1 = {"W1": tf.Variable(tf.truncated_normal([NODES_INPUT, NODES_H1],
                                                        stddev=0.0842,
                                                        dtype=tf.float64),
                                                        name='W1'),
                 "b1": tf.constant(0.1, shape=[NODES_H1], dtype=tf.float64, name='b1')}

hidden_layer2 = {"W2": tf.Variable(tf.truncated_normal([NODES_H1, NODES_H2],
                                                        stddev=0.0842,
                                                        dtype=tf.float64),
                                                        name='W2'),
                 "b2": tf.constant(0.1, shape=[NODES_H2], dtype=tf.float64, name='b2')}


# Capa de salida
output_layer = {"W_out": tf.Variable(tf.truncated_normal([NODES_H2, NODES_OUPUT],
                                                        stddev=0.0842,
                                                        dtype=tf.float64),
                                                        name='W_output'),
                "b_out": tf.constant(0.1, shape=[NODES_OUPUT], dtype=tf.float64, name='b_output')}


# Se crean histogramas para mostrar en TensorBoard
tf.summary.histogram("weight_1", hidden_layer1["W1"])
tf.summary.histogram("weight_2", hidden_layer2["W2"])
tf.summary.histogram("weight_out", output_layer["W_out"])


# In[21]:


print(hidden_layer1)


# In[22]:


# Calcular la salida de la 1er capa oculta
# h(x) = x*w + bias
hidden_layer1_output = tf.add(tf.matmul(es_vectores, hidden_layer1["W1"]), hidden_layer1["b1"])
# https://www.learnopencv.com/understanding-autoencoders-using-tensorflow-python/
# 3.4 Why do we use a leaky ReLU and not a ReLU as an activation function?
# We want gradients to flow while we backpropagate through the network. 
# We stack many layers in a system in which there are some neurons 
# whose value drop to zero or become negative. Using a ReLU as an activation 
# function clips the negative values to zero and in the backward pass, 
# the gradients do not flow through those neurons where the values become zero. 
# Because of this the weights do not get updated, and the network stops learning 
# for those values. So using ReLU is not always a good idea. However, we encourage 
# you to change the activation function to ReLU and see the difference.
# Parámetro para función de activación leaky_relu
alpha = tf.constant(0.2, dtype=tf.float64)
# Función de activación usando leaky_relu
tf.summary.histogram("pre_activations_h1", hidden_layer1_output)
hidden_layer1_output = tf.nn.leaky_relu(hidden_layer1_output,alpha, name="h1Activation")
tf.summary.histogram('activationsh1', hidden_layer1_output)


# In[23]:


# Calcular la salida de la 2da capa oculta
# h(x) = x*w + bias
hidden_layer2_output = tf.add(tf.matmul(hidden_layer1_output, hidden_layer2["W2"]), hidden_layer2["b2"])
# Función de activación usando leaky_relu
tf.summary.histogram("pre_activations_h2", hidden_layer2_output)
hidden_layer2_output = tf.nn.leaky_relu(hidden_layer2_output,alpha, name="h2Activation")
tf.summary.histogram('activationsh2', hidden_layer2_output)


# In[24]:


# calcular la salida la NN
output = tf.add(tf.matmul(hidden_layer2_output,output_layer["W_out"]), output_layer["b_out"])
# Función de activación usando softmax
tf.summary.histogram("pre_activations_output", output)
nah_predicted = tf.nn.softmax(output, name="outActivation")
tf.summary.histogram('activationsout', nah_predicted)
print(nah_predicted.shape, y.shape)


# In[25]:


# Función de error (Mean Square Error)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=nah_predicted))
#loss = tf.reduce_mean(tf.reduce_sum((nah_predicted - y) ** 2))
loss = tf.reduce_mean(tf.squared_difference(nah_predicted, y), name="loss_f")
tf.summary.scalar("loss", loss)


# In[26]:

# Al parecer se presenta el problema de vanishing gradient 
# problem (https://medium.com/@anishsingh20/the-vanishing-gradient-problem-48ae7f501257) 
# por lo que se utiliza la función tf.clip_by_global_norm para evitar que el 
# gradiente se vuelva muy pequeño y tarde más en converger.
# https://www.tensorflow.org/versions/r0.12/api_docs/python/train/gradient_clipping

# Create an optimizer.
# En pruebas la función tf.train.AdamOptimizer parace quedarse estancada 
# en un mínimo local al momento de minimizar el error.
optimiser = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE,
                                      name="AdagradOptimizer")

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

# Accuracy 
with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(nah_predicted, 1), tf.argmax(y, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
tf.summary.scalar('accuracy', accuracy)


# In[27]:


LOGPATH = "./logs/NN_MSE_ADAG_" + str(NODES_H1) + "h1_ " + str(NODES_H2) + "h2" 
print("logpath", LOGPATH)
#Session
config = tf.ConfigProto(intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=1,
                        #log_device_placement=True
                        )
sess = tf.Session(config=config)
# Initialize variables
summaryMerged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGPATH, sess.graph)
init = tf.global_variables_initializer()
sess.run(init)


# In[28]:


tmp_hidden_layer1, tmp_hidden_layer2 = sess.run(hidden_layer1), sess.run(hidden_layer2)


# In[29]:


for i in range(EPOCHS):
    '''
    offset = (step * BATCH_SIZE) % (es_vectores.shape[0] - BATCH_SIZE)
    batch_data = es_vectores[offset:(offset + BATCH_SIZE), :]
    batch_target = na_vectores[offset:(offset + BATCH_SIZE), :]
    '''

    _loss, _, sumOut = sess.run([loss, train_op, summaryMerged], feed_dict={
                                X: es_vectores, y: na_vectores})
    if (i % 500) == 0:
        print("Epoch:",i,"/",EPOCHS, "\tLoss:", _loss)
    writer.add_summary(sumOut, i)

    if i % 100 == 99:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _loss, _, sumOut = sess.run([loss, train_op, summaryMerged], feed_dict={
                                    X: es_vectores, y: na_vectores},
                                    options=run_options,
                                    run_metadata=run_metadata)
        writer.add_run_metadata(run_metadata, 'step%03d' % i)
        writer.add_summary(sumOut, i)

writer.close()


# In[30]:


print(sess.run(hidden_layer1))


# In[31]:


tmp_hidden_layer1["W1"] - sess.run(hidden_layer1["W1"])


# In[32]:


sess.run(hidden_layer2)
