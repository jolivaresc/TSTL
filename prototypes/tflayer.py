import tensorflow as tf

tf.set_random_seed(42)
tf.reset_default_graph()
print(tf.test.gpu_device_name())
LEARNING_RATE = 1
EPOCHS = 1000
# Dimensión de vectores de entrada (número de neuronas en capa de entrada).
NODES_INPUT = es_vectores[0].size

# Número de neuronas en capas ocultas.
NODES_H1 = 300  # 70 - 20 - 15
NODES_H2 = 240  # 42 - 20
NODES_H3 = 220  # 42 - 20
NODES_H4 = 200  # 42 - 20
NODES_OUTPUT = na_vectores[0].size

# Inicializar pesos usando xavier_init
XAVIER_INIT = True


with tf.name_scope('input'):
    # El valor None indica que se puede modificar la dimensión de los tensores
    # por si se usan todos los vectores o batches.
    X = tf.placeholder(shape=[None, NODES_INPUT],
                       dtype=tf.float64, name='input_es')
    y = tf.placeholder(shape=[None, NODES_OUTPUT],
                       dtype=tf.float64, name='target_na')

fc1 = tf.layers.dense(X,NODES_H1,activation=tf.nn.relu)
cost = tf.reduce_mean((fc1-y)**2)
optimizer = tf.train.AdagradOptimizer(LR).minimize(cost)

