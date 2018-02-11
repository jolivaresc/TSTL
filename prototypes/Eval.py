
# coding: utf-8

# In[1]:


import tensorflow as tf
import utils
import numpy as np

__author__ = "Olivares Castillo José Luis"
tf.__version__


# In[2]:


es,na = utils.load_embeddings()


# In[3]:


es.shape[1]


# In[4]:


index_es, index_na = utils.get_seed_index(es, na)
es_vectores = utils.get_vectors(es, index_es)
na_vectores = utils.get_vectors(na, index_na)


# In[5]:


#index_es


# In[6]:


lexes=[es.iloc[_][0] for _ in index_es]
lexna=[na.iloc[_][0] for _ in index_na]


# In[7]:


new_lexicon =list(zip(lexes,lexna))


# In[8]:


#for _ in new_lexicon:
#    print(_[0],_[1])

with open("newlexicon.lst","w") as fd:
    for _ in new_lexicon:
        fd.write(_[0]+" "+_[1]+"\n")
# In[9]:


es[es[0] == "y"][0]


# In[10]:


es[es[0] == "se"][0]


# In[11]:


na[na[0] == "no"][0]


# In[12]:


es_dummy = es.drop(es.columns[0],axis=1)
na_dummy = na.drop(na.columns[0],axis=1)
es_vectores1 = np.array(es_dummy)
na_vectores1 = np.array(na_dummy)
print(na_vectores1.shape)


# In[13]:


#es_vectores = [es.iloc[_].loc[1::] for _ in range(es.shape[0])]


# In[14]:


print(es.iloc[2537][0])


# In[15]:


# Palabra y vector
#indice = 1756
#print(es.iloc[3508][0])
test_vectors = np.array([np.array(es.iloc[indice][1::]).astype(np.float64) for indice in index_es])
#test_vectors = na_vectores1
#test_vectors = np.array(es.iloc[2537][1::]).astype(np.float64)
#test_vectors.resize(1,128)
print(test_vectors.shape)


# In[16]:


sess = tf.Session()


# In[17]:


saver = tf.train.import_meta_graph('./models/model1/Adagrad_H_300_LR_0.4666.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./models/model1/'))


# In[18]:


graph = tf.get_default_graph()


# In[19]:


#print(sess.run("output/W:0"))

output_W = graph.get_tensor_by_name("output/W:0")
output_b = graph.get_tensor_by_name("output/b:0")
output_xwb = graph.get_tensor_by_name("output/xw_plus_b:0")
# In[20]:


X = graph.get_tensor_by_name("input/input_es:0")
#y = graph.get_tensor_by_name("input/target_na:0")


# In[21]:


print(X.shape)


# In[22]:


feed_dict = {X:test_vectors}


# In[23]:


output_NN = graph.get_tensor_by_name("output_1:0")


# In[24]:


pred = sess.run(output_NN,feed_dict)
print (type(pred[0]),pred.shape)


# In[25]:


#print(pred[0].shape)


# In[26]:


top_10=[utils.get_top10_closest(pred[_],na_vectores1) for _ in range(pred.shape[0])]
#top_10=utils.get_top10_closest(pred[0],na_vectores1)


# In[27]:


closest = [utils.get_closest_words_to(top_10[_],na) for _ in range(pred.shape[0])]
#closest = utils.get_closest_words_to(top_10,na)


# In[28]:


closest[0]


# In[29]:


#acc1=[(_,lexna[_] == closest[_][0]) for _ in range(pred.shape[0])]
acc=[(lexna[_] == closest[_][0]) for _ in range(pred.shape[0])]


# In[30]:


#acc


# In[31]:


precision_list = np.array(acc).astype(np.int)
precision_list


# In[32]:


from collections import Counter


# In[33]:


precision = Counter(precision_list)


# In[34]:


precision


# In[35]:


print("Precisión con lexicon de entrenamient:",precision[1]/len(precision))

