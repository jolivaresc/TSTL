
# coding: utf-8

# 
# 
# a = tf.constant(4)
# b = tf.constant(13)
# 
# c = tf.multiply(a,b)
# print(c)
# 
# with tf.Session() as sess:
#     result = sess.run(c)
#     print(result)

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


SEED = 42
tf.set_random_seed(SEED)

# MNIST dataset
mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

# Número de nodos en las capas ocultas.
hidden_nodes_1 = 400
hidden_nodes_2 = 300
hidden_nodes_3 = 300

#Etiquetas
n_classes = 10
batch_size = 100

# Inputs
x = tf.placeholder('float',[None,784])
# labels
y = tf.placeholder('float')



# In[2]:

def NN(data):
    
    '''
    Se inicializan los pesos y bias de las capas ocultas de la red neuronal con valores aleatorios. Los
    pesos y bias (weights, bias) son tensores y se almacenan en un diccionario que corresponden a cada capa.    
    '''
    hidden_l1 = {'weights': tf.Variable(tf.truncated_normal([784,hidden_nodes_1], stddev=0.1)),
                 'biases': tf.truncated_normal([hidden_nodes_1], stddev=0.1)}
    
    hidden_l2 = {'weights': tf.Variable(tf.truncated_normal([hidden_nodes_1,hidden_nodes_2], stddev=0.1)),
                 'biases': tf.Variable(tf.truncated_normal([hidden_nodes_2], stddev=0.1))}
    
    hidden_l3 = {'weights': tf.Variable(tf.truncated_normal([hidden_nodes_2,hidden_nodes_3], stddev=0.1)),
                 'biases': tf.Variable(tf.truncated_normal([hidden_nodes_3], stddev=0.1))}
    
    output_layer = {'weights': tf.Variable(tf.truncated_normal([hidden_nodes_3,n_classes], stddev=0.1)),
                    'biases': tf.Variable(tf.truncated_normal([n_classes], stddev=0.1))}
    
    # (input * weigths) + bias
    layer1 = tf.add(tf.matmul(data,hidden_l1['weights']),hidden_l1['biases'])
    layer1 = tf.nn.relu(layer1) # Función de activación
    
    layer2 = tf.add(tf.matmul(layer1,hidden_l2['weights']),hidden_l2['biases'])
    layer2 = tf.nn.relu(layer2) # Función de activación
    
    layer3 = tf.add(tf.matmul(layer2,hidden_l3['weights']),hidden_l3['biases'])
    # Función de activación
    layer3 = tf.nn.relu(layer3)
    
    output = tf.add(tf.matmul(layer3,output_layer['weights']),output_layer['biases'])
    
    return layer1,layer2,layer3,output
    


# In[3]:

def train(x):
    l1,l2,l3,prediction = NN(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    no_epochs = 5
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(no_epochs):
            loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                loss += c
            print('Epoch: ',epoch, 'completed out of',no_epochs,'loss:',loss)
            pass
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print('---------')
    
    return l1,l2,l3,prediction


# In[4]:

l1,l2,l3,prediction = train(x)


# In[5]:

sess.run(l1)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



