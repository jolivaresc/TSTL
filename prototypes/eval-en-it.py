import tensorflow as tf
import pandas as pd
import numpy as np
import utils
from collections import defaultdict

__author__ = "Olivares Castillo José Luis"


def read(file, threshold=0, vocabulary=None, dtype='float'):
    # Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
    # https://github.com/artetxem/vecmap/blob/master/embeddings.py
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(
        threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


def closest_word_to(top_10, words):
    return [words[index] for index, _ in top_10]


def read_lexicon(source):
    src, trg = list(), list()
    if source.__eq__("train"):
        with open("../dataset/OPUS_en_it_europarl_train_5K.txt", "r", encoding='utf-8') as file:
            for line in file:
                src.append(line.split()[0])
                trg.append(line.split()[1])
        return (src, trg)
    elif source.__eq__("test"):
        with open("../dataset/OPUS_en_it_europarl_test.txt", "r", encoding='utf-8') as file:
            for line in file:
                src.append(line.split()[0])
                trg.append(line.split()[1])
        return (src, trg)


def get_vectors(lexicon, words, embeddings, dtype='float'):
    matrix = np.empty((len(lexicon), embeddings.shape[1]), dtype=dtype)
    for i in range(len(lexicon)):
        if lexicon[i] in words:
            matrix[i] = embeddings[words.index(lexicon[i])]
    return np.asarray(matrix, dtype=dtype)


en, it = read_lexicon("test")
print(len(en), len(it))

source_vec = open("../dataset/EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt",
                  encoding="utf-8", errors="surrogateescape")
words_en, en_vec = read(source_vec)
eval_en = list(set(en))
src_vec = get_vectors(eval_en, words_en, en_vec)
print(src_vec.shape)

target_vec = open("../dataset/IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt",
                  encoding="utf-8", errors="surrogateescape")
words_it, it_vec = read(target_vec)
eval_it = list(set(it))
trg_vec = get_vectors(eval_it, words_it, it_vec)
print(trg_vec.shape)


###############


test_vectors = src_vec

sess = tf.Session()
saver = tf.train.import_meta_graph(
    './models/model1111_gpu/model2250.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models/model1111_gpu/'))


graph = tf.get_default_graph()
X = graph.get_tensor_by_name("input/input_es:0")

kprob = graph.get_tensor_by_name("dropout_prob:0")

#print([n.name for n in tf.get_default_graph().as_graph_def().node])
output_NN = graph.get_tensor_by_name("xw_plus_b_1:0")


feed_dict = {X: test_vectors, kprob: 1}
pred = sess.run(output_NN, feed_dict)


top_10 = [utils.get_top10_closest(pred[_], it_vec)
          for _ in range(pred.shape[0])]

closest = [closest_word_to(top_10[_], words_it)for _ in range(pred.shape[0])]


resultados = {palabra_en: top_10_it
              for (palabra_en, top_10_it) in zip(eval_en, closest)}


pares_eval = list(zip(en, it))
gold = defaultdict(list)

for palabra_en, palabra_it in pares_eval:
    gold[palabra_en].append(palabra_it)
gold = dict(gold)


p1, p5, p10 = 0, 0, 0
list_en_eval = list(resultados.keys())
hits, not_found = [], []

for palabra_gold in list_en_eval:
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

length = list_en_eval.__len__()
print("not found:", not_found.__len__(),
      "-", not_found.__len__() / length, "%")
print("P@1:", p1, "\tP@5:", p5, "\tP@10:", p10)
print("P@1:", p1 / length, "\tP@5:", p5 /
      length, "\tP@10:", p10 / length)
