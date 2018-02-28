# coding: utf-8
"""
- Para ejecutar: python evalset.py SOURCE_VEC SOURCE_LEXICON

- SOURCE_VEC es el archivo a leer que contiene vectores 
  [ n2v | w2v3 | w2v7 | w2v14 ]
- SOURCE_LEXICON es el archivo del lexicon a utilizar [seed | eval]
"""

import tensorflow as tf
import utils
import pandas as pd
import numpy as np
from sys import argv
from collections import defaultdict


__author__ = "Olivares Castillo José Luis"


"""
- Se carga archivo con vectores [N2V/W2V] en dataframes
- Se carga archivo lexicon [entrenamiento/evaluación]
- Se obtienen las palabras del lexicon.
- Se obtienen índices de las palabras a traducir de los dataframes [N2V/W2V]
"""
es, na = utils.load_embeddings(argv[1])
na_dummy = na.drop(na.columns[0], axis=1)
na_vectores1 = na_dummy.values.astype(np.float64)
eval_set = utils.get_lexicon(argv[2])
eval_es = list(set(eval_set["esp"]))
eval_es_index = [int(es[es[0] == palabra].index[0])
                 for palabra in eval_es]

PRINT = False

"""
- Se buscan en los dataframes [N2V/W2V] los índices de las palabras a traducir para obtener sus representaciones vectoriales.
- Se crea un arreglo matricial con las representaciones vectoriales obtenidas
"""
eval_es_vectores = utils.get_vectors(es, eval_es_index)
test_vectors = np.array([np.array(es.iloc[indice][1::]).astype(
    np.float64) for indice in eval_es_index])

"""
- Se inicia una sesión de TensorFlow.
- Se carga el modelo a evaluar.
- Se restaura el modelo usando la sesión de TF.
"""
sess = tf.Session()
saver = tf.train.import_meta_graph('./models/model1111_gpu/model2250.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models/model1111_gpu/'))
#saver = tf.train.import_meta_graph('./models/model_klein/modelklein.ckpt.meta')
#saver.restore(sess, tf.train.latest_checkpoint('./models/model_klein/'))
#saver = tf.train.import_meta_graph('./models/model_joyce/modeljoyce.ckpt.meta')
#saver.restore(sess, tf.train.latest_checkpoint('./models/model_joyce/'))

"""
- Se obtiene el grafo del modelo restaurado.
- Se guardan en variables los tensores necesarios para realizar la evaluación [inputs]
"""
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("input/input_es:0")
#y = graph.get_tensor_by_name("input/target_na:0")
kprob = graph.get_tensor_by_name("dropout_prob:0")

#print([n.name for n in tf.get_default_graph().as_graph_def().node])

# output_NN = graph.get_tensor_by_name("output/xw_plus_b:0")#model1937
"""
- Se guarda la salida del modelo en una variable, será la que nos dará el resultado de evualuar el modelo
- NOTA: La salida varía de acuerdo al número de capas que tenga la red.
"""
output_NN = graph.get_tensor_by_name("xw_plus_b_5:0")
#output_NN = graph.get_tensor_by_name("nah_predicted:0")
#output_NN = graph.get_tensor_by_name("dense_2/BiasAdd:0")
#output_NN = graph.get_tensor_by_name("output_1:0")


"""
- Se carga la matriz de vectores a los tensores de entrada del modelo.
- Se evalua el modelo y el resultado de las predicciones se almacenan en una matriz.
"""
feed_dict = {X: test_vectors, kprob: 1}
pred = sess.run(output_NN, feed_dict)
#print (type(pred[0]),pred.shape)


"""
- Se obtienen los 10 vectores/palabras más cercanos a cada predicción obtenida por el modelo
"""
top_10 = [utils.get_top10_closest(pred[_], na_vectores1)
          for _ in range(pred.shape[0])]
closest = [utils.get_closest_words_to(top_10[_], na)
           for _ in range(pred.shape[0])]


"""
- Se crea un diccionario con la palabra y las predicciones que obtuvo el modelo
"""
resultados = {palabra_es: top_10_nah for (
    palabra_es, top_10_nah) in zip(eval_es, closest)}


"""
- Se crea un diccionario con la palabra y su traducción real [gold standard]
"""
esp = list(eval_set["esp"].values)
nah = list(eval_set["nah"].values)
pares_eval = list(zip(esp, nah))
gold = defaultdict(list)
for palabra_es, palabra_na in pares_eval:
    gold[palabra_es].append(palabra_na)
gold = dict(gold)


"""
- Se evalua la precisión de las predicciones usando P@k usando una lista de hits y sus índices.
- Se guardan en una lista las palabras que no fueron encontradas dentro de las traducciones predichas.
"""
p1, p5, p10 = 0, 0, 0
list_esp_eval = list(resultados.keys())
hits, not_found = list(), list()

# Se buscan las traducciones gold standard dentro de las predicciones y se obtiene
# P@K, sino se encuentran, se añade a una lista de no encontrados.
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
        if min(hits) > 5 and min(hits) < 10:
            p10 += 1
    else:
        not_found.append(palabra_gold)
    hits.clear()

length = list_esp_eval.__len__()
print("not found:", not_found.__len__(),
      "-", not_found.__len__() / length, "%")
print("P@1:", p1, "\tP@5:", p5, "\tP@10:", p10)
print("P@1:", p1 / length, "\tP@5:", p5 / length, "\tP@10:", p10 / length)

if PRINT:
    # Diccionario que contiene la palabra, sus traducción y los canditados a
    # traducción obtenidos por el autoencoder
    resultados_gold = dict()
    for k, v in resultados.items():
        resultados_gold[k] = {"GOLD": gold[k], "RESULTS": v}

    # Muestra palabras que no se encontraron en los candidatos a traducción..
    print("\n=================================================")
    print("PALABRAS NO ENCONTRADAS Y SUS CANDIDATOS...")
    for palabra in not_found:
        print(palabra.upper() + ":", "\nGOLD", resultados_gold[palabra]["GOLD"],
              "\nRESULTADOS", resultados_gold[palabra]["RESULTS"], end="\n" * 2)

    # Muestra las palabras, su traducción y los canditados a traducción
    print("\n=================================================")
    print("PALABRAS DEL EVALSET CON SU TRADUCCIÓN Y 10 CANDIDATOS")
    for k, v in resultados_gold.items():
        print("Palabra:", k.upper(), "\nGOLD:", v["GOLD"], "\nRESULTADOS:",
              v["RESULTS"], end="\n" * 2)
