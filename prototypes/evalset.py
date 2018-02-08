
# coding: utf-8

# In[1]:


import tensorflow as tf
import utils
import pandas as pd
import numpy as np
__author__ = "Olivares Castillo José Luis"

es,na = utils.load_node2vec()
na_dummy = na.drop(na.columns[0],axis=1)
na_vectores1 = np.array(na_dummy)
eval_set = pd.read_csv("../lexiconevaluacion/evaluationset",delimiter=" ",names=["esp","nah"])
eval_es = list(set(eval_set["esp"]))
eval_es_index = [int(es[es[0] == palabra].index[0])
                  for palabra in eval_es]



eval_es_vectores = utils.get_vectors(es,eval_es_index)
test_vectors = np.array([np.array(es.iloc[indice][1::]).astype(np.float64) for indice in eval_es_index])



sess = tf.Session()
saver = tf.train.import_meta_graph('./models/model1111_gpu/model2250.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./models/model1111_gpu/'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("input/input_es:0")
#y = graph.get_tensor_by_name("input/target_na:0")


#print([n.name for n in tf.get_default_graph().as_graph_def().node])

#output_NN = graph.get_tensor_by_name("output/xw_plus_b:0")#model1937
output_NN = graph.get_tensor_by_name("xw_plus_b_1:0")
#output_NN = graph.get_tensor_by_name("dense_2/BiasAdd:0")
#output_NN = graph.get_tensor_by_name("output_1:0")
feed_dict = {X:test_vectors}
pred = sess.run(output_NN,feed_dict)
print (type(pred[0]),pred.shape)

top_10=[utils.get_top10_closest(pred[_],na_vectores1) for _ in range(pred.shape[0])]
closest = [utils.get_closest_words_to(top_10[_],na) for _ in range(pred.shape[0])]


es[es[0]=="adquisición"].index

resultados = {palabra_es:top_10_nah for (palabra_es,top_10_nah) in zip(eval_es,closest)}
esp = list(eval_set["esp"].values)
nah = list(eval_set["nah"].values)
pares_eval = list(zip(esp,nah))
from collections import defaultdict
gold = defaultdict(list)
for palabra_es,palabra_na in pares_eval:
    gold[palabra_es].append(palabra_na)

gold = dict(gold)

p1 = 0
p5 = 0
p10 = 0
list_esp_eval = (list(resultados.keys()))
hits=list()

not_found = list()


for palabra_gold in list_esp_eval:
    for i in gold[palabra_gold]:
        if i in resultados[palabra_gold]:
            hits.append(resultados[palabra_gold].index(i))
    if hits.__len__() > 0:
        if min(hits) == 0:
            p1 += 1
            p5 += 1
            p10 += 1
        if min(hits) >= 1 and min(hits) <= 5:
            p5 += 1
            p10 += 1
        if min(hits) > 5 and min(hits) <= 10:
            p10 += 1
        #print(palabra_gold,min(hits),hits,p1,p5,p10)
    else:
        not_found.append(palabra_gold)
        #print(palabra_gold+": NOT FOUND")
   
    hits.clear()

length=list_esp_eval.__len__()
print("\nnot found:", not_found.__len__(), "\nP@1:", p1 / length,
      "\tP@5:", p5 / length, "\tP@10:", p10 / length)
     
