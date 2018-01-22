#https://beckernick.github.io/neural-network-scratch/

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
#%matplotlib inline

# %%
np.random.seed(42)
num_observations = 7000
NUM_OBS = 500 * 3
TMP = int(NUM_OBS / 3)
INPUT_DIM = 2
OFFSET = 5
#%%
x1 = np.random.multivariate_normal([0, 0], [[2, .75],[.75, 2]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
x3 = np.random.multivariate_normal([2, 8], [[0, .75],[.75, 0]], num_observations)


# concatena todos los valores en un solo vector
#simulated_separableish_features = np.vstack((x1, x2, x3)).astype(np.float32)
# vector español
es = np.vstack((x1, x2, x3)).astype(np.float32)
#es = es.astype(np.float32)

# simulated_labels = np.hstack((np.zeros(num_observations),
#                             np.ones(num_observations),
#                             np.ones(num_observations) + 1))

na = np.hstack((np.zeros(num_observations),
                np.ones(num_observations),
                np.ones(num_observations) + 1))


#%%
# PLOT
plt.figure(figsize=(12,8))
plt.scatter(es[:, 0], es[:, 1],c = na, alpha = None)
plt.grid()

#%%

labels_onehot = np.zeros((na.shape[0], 3)).astype(int)
labels_onehot[np.arange(len(na)), na.astype(int)] = 1

X_train, X_test,y_train, y_test = train_test_split(
                        es,
                        labels_onehot,
                        test_size = .1,
                        random_state = 42)


#%%
hidden1_nodes = 30
hidden2_nodes = 10
num_labels = y_train.shape[1]
num_labels

batch_size = 500
num_features = X_train.shape[1]
learning_rate = .001

graph = tf.Graph()
with graph.as_default():

    # Data
    tf_X_train = tf.placeholder(tf.float32, shape = [None, num_features])
    tf_y_train = tf.placeholder(tf.float32, shape = [None, num_labels])
    tf_X_test = tf.constant(X_test)

    # Weights and Biases
    layer1_weights = tf.Variable(tf.truncated_normal([num_features, hidden1_nodes]))
    layer1_biases = tf.Variable(tf.random_normal([hidden1_nodes]))

    layer2_weights = tf.Variable(tf.truncated_normal([hidden1_nodes, num_labels]))
    layer2_biases = tf.Variable(tf.random_normal([num_labels]))

    # No hay mejoras importantes si se añade una capa extra
    layer3_weights = tf.Variable(tf.truncated_normal([hidden2_nodes, num_labels]))
    layer3_biases = tf.Variable(tf.random_normal([num_labels]))
    # Four-Layer Network
    def four_layer_network(data):
        input_layer = tf.add(tf.matmul(data, layer1_weights), layer1_biases)
        hidden1 = tf.nn.relu(input_layer)
        # hidden2 = tf.add(tf.matmul(hidden1, layer2_weights), layer2_biases)
        # hidden2 = tf.nn.relu(hidden2)
        output_layer = tf.add(tf.matmul(hidden1, layer2_weights), layer2_biases)
        return output_layer

    # Model Scores
    model_scores = four_layer_network(tf_X_train)

    # Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_scores,
                            labels=tf_y_train))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions
    train_prediction = tf.nn.softmax(model_scores)
    test_prediction = tf.nn.softmax(four_layer_network(tf_X_test))
#%%
def accuracy(predictions, labels):
    preds_correct_boolean =  np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy
#%%
num_steps = 10001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        minibatch_data = X_train[offset:(offset + batch_size), :]
        minibatch_labels = y_train[offset:(offset + batch_size)]

        feed_dict = {tf_X_train : minibatch_data, tf_y_train : minibatch_labels}

        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict = feed_dict
            )

        if step % 1000 == 0:
            print('Minibatch loss at step {0}: {1}'.format(step, l))

    print('Test accuracy: {0}%'.format(accuracy(test_prediction.eval(), y_test)))

#%%
