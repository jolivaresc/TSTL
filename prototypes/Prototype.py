
# coding: utf-8

# In[1]:


#http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/deep-neural-nets/
#http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/
#https://jhui.github.io/2017/03/12/TensorBoard-visualize-your-learning/
#https://thecodacus.com/tensorboard-tutorial-visualize-networks-graphically/#.WlVvmnWnG00
#http://ruder.io/transfer-learning/

import tensorflow as tf
from numpy import array, float32
from pandas import read_csv, set_option

# In[2]:


tf.set_random_seed(42)


# In[3]:


set_option('display.max_colwidth', -1)
set_option('precision', 18)


# In[4]:


es = read_csv("../vectors/es.node2vec.embeddings", delimiter = " ", skiprows = 1, header = None)
na = read_csv("../vectors/na.node2vec.embeddings", delimiter = " ", skiprows = 1, header = None)


# In[5]:


es.head()


# In[6]:


na.head()


# In[7]:


es_vectores = array(es[:1900].loc[:,1::]).astype(float32)
na_vectores = array(na[:1900].loc[:,1::]).astype(float32)


# In[8]:


es_vectores


# In[9]:


na_vectores


# In[10]:


es_palabras = es[0]
na_palabras = na[0]


# In[11]:


print("shape es:",es_vectores.shape,"\nshape na:",na_vectores.shape)


# # Hyperparameters

# In[12]:


# Hyperparameters
LEARNING_RATE = 0.3
EPOCHS = 10
BATCH_SIZE = 100

NODES_INPUT = es_vectores[0].size
NODES_H1 = 100
NODES_H2 = 100
NODES_OUPUT = na_vectores[0].size
INSTANCES = es_vectores.__len__()
NUM_STEPS = 30000


# In[13]:


X = tf.placeholder(shape=[None,NODES_INPUT], dtype=tf.float32,name='input_es')
y = tf.placeholder(shape=[None,NODES_OUPUT], dtype=tf.float32,name='target_na')
print("X:",X.shape,"y:",y.shape)


# In[14]:


# Capas ocultas (weights & bias)
hidden_layer1 = {"W1":tf.Variable(tf.truncated_normal([NODES_INPUT,NODES_H1],stddev=0.01),name = 'W1'),
                 "b1":tf.constant(0.1,shape=[NODES_H1], name = 'b1')}

hidden_layer2 = {"W2":tf.Variable(tf.truncated_normal([NODES_H1,NODES_H2],stddev=0.01),name = 'W2'),
                 "b2":tf.constant(0.1,shape=[NODES_H2], name = 'b2')}

# Capa de salida
output_layer = {"W_out":tf.Variable(tf.truncated_normal([NODES_H2,NODES_OUPUT],stddev=0.01),
                name = 'W_output'),
                "b_out":tf.constant(0.1,shape=[NODES_OUPUT], name = 'b_output')}


tf.summary.histogram("weight_1", hidden_layer1["W1"])
tf.summary.histogram("weight_2", hidden_layer2["W2"])
tf.summary.histogram("weight_out", output_layer["W_out"])


# In[15]:


print(hidden_layer1)


# In[16]:


# Calcular la salida de la 1er capa oculta
# h(x) = x*w + bias
hidden_layer1_output = tf.add(tf.matmul(es_vectores, hidden_layer1["W1"]), hidden_layer1["b1"])
# Función de activación usando ReLU
tf.summary.histogram("pre_activations_h1", hidden_layer1_output)
hidden_layer1_output = tf.nn.relu(hidden_layer1_output,name="h1Activation")
tf.summary.histogram('activationsh1', hidden_layer1_output)


# In[17]:


# Calcular la salida de la 2da capa oculta
# h(x) = x*w + bias
hidden_layer2_output = tf.add(tf.matmul(hidden_layer1_output, hidden_layer2["W2"]), hidden_layer2["b2"])
# Función de activación usando ReLU
tf.summary.histogram("pre_activations_h2", hidden_layer2_output)
hidden_layer2_output = tf.nn.relu(hidden_layer2_output,name="h2Activation")
tf.summary.histogram('activationsh2', hidden_layer2_output)


# In[18]:


# calcular la salida la NN
output = tf.add(tf.matmul(hidden_layer2_output, output_layer["W_out"]),output_layer["b_out"])
# Función de activación usando softmax
tf.summary.histogram("pre_activations_output", output)
nah_predicted = tf.nn.softmax(output,name="outActivation")
tf.summary.histogram('activationsout', nah_predicted)
print(nah_predicted.shape,y.shape)


# In[19]:


# Función de error (Mean Square Error)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=nah_predicted))
loss = tf.reduce_mean(tf.squared_difference(nah_predicted,y),name="loss_f")
#loss = tf.reduce_mean(tf.reduce_sum((nah_predicted - y) ** 2))
tf.summary.scalar("cost",loss)


# In[20]:


# optimiser, 
optimiser = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE,
                                      name="AdagradOptimizer").minimize(loss, name="loss")


# In[21]:


#Session
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 44
config.inter_op_parallelism_threads = 44
sess = tf.Session(config=config)
# Initialize variables
summaryMerged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/NN",sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


# In[22]:


tmp_hidden_layer1,tmp_hidden_layer2 = sess.run(hidden_layer1),sess.run(hidden_layer2)


# In[23]:


for i in range(NUM_STEPS):
    '''
    offset = (step * BATCH_SIZE) % (es_vectores.shape[0] - BATCH_SIZE)
    batch_data = es_vectores[offset:(offset + BATCH_SIZE), :]
    batch_target = na_vectores[offset:(offset + BATCH_SIZE), :]
    '''
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    _loss,_, sumOut = sess.run([loss,optimiser, summaryMerged], feed_dict={
                        X: es_vectores, y: na_vectores},
                        options=run_options,
                        run_metadata=run_metadata)
    if (i % 500) == 0:
        print(_loss)
    writer.add_summary(sumOut, i)
    writer.add_run_metadata(run_metadata, 'step%03d' % i)
writer.close()


# In[24]:


print(sess.run(hidden_layer1))


# In[25]:



tmp_hidden_layer1["W1"]-sess.run(hidden_layer1["W1"])


# In[26]:


sess.run(hidden_layer2)


sess.close()
