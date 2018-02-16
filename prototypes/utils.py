
# coding: utf-8

from numpy import dot, float64, array, sqrt, matmul, einsum
from numpy.linalg import norm
from pandas import set_option, read_csv, DataFrame

__author__ = "Olivares Castillo José Luis"


# Se establece la precisión con la que Pandas lee el archivo para evitar que
# trunque el valor de los vectores
set_option('display.max_colwidth', -1)
set_option('precision', 18)


def load_embeddings(source):
    """Esta función lee los archivos para almacenar los vectores node2vec del español
    y náhuatl los retorna en dataframes de Pandas.

    * Arguments:
        `source` {string} -- Archivo a leer.
    * Returns:
        Dataframe (2) -- dataframes con palabras - node2vec.
    """

    # Cargar embeddings desde archivos
    if source.__eq__("n2v"):
        esp = open("../vectors/es.node2vec.embeddings", "r")
        nah = open("../vectors/na.node2vec.embeddings", "r")
    elif source.__eq__("w2v3"):
        esp = open("../vectors/es.w2v.300", "r")
        nah = open("../vectors/na.w2v.300", "r")
    elif source.__eq__("w2v7"):
        esp = open("../vectors/es.w2v.700", "r")
        nah = open("../vectors/na.w2v.700", "r")
    elif source.__eq__("w2v14"):
        esp = open("../vectors/es.w2v.1400", "r")
        nah = open("../vectors/na.w2v.1400", "r")

    # Se guardan los archivos en listas.
    lines_es = esp.readlines()
    lines_na = nah.readlines()
    
    # Se elimina el primer elemento de la lista ya que no se utiliza
    lines_es.pop(0)
    lines_na.pop(0)

    # Listas temporales
    tmp_es, tmp_na = list(), list()
    
    # Para cada item de la lista, se separa por espacios y se añade a la lista temporal
    for line in lines_es:
        tmp_es.append(line.split())

    for line in lines_na:
        tmp_na.append(line.split())

    # Se eliminan variables que ya no se utilizan
    del lines_es
    del lines_na
    del esp
    del nah


    # Variables de retorno
    es_df = DataFrame.from_records(tmp_es)
    na_df = DataFrame.from_records(tmp_na)
    # Se regresan los dataframes con los embeddings.
    return es_df, na_df


def get_lexicon(source):
    """Función para leer lexicon.
    
    Arguments:
        source {string} -- Lexicon a leer
    
    Returns:
        Pandas.dataframe -- Dataframe del lexicon
    """
    if source.__eq__("eval"):
        return read_csv("../lexiconevaluacion/evaluationset", delimiter=" ", names=["esp", "nah"])
    elif source.__eq__("seed"):
        return read_csv("../lexiconessemilla/newlexiconb.lst", delimiter=" ", names=["esp", "nah"])
    else:
        print("ERR: Ingrese un lexicon válido..")


def get_dataframe_index(dataframe, palabra):
    """Obtiene el índice dentro del dataframe de la palabra si es que
    se existe, sino retorna falso.

    Arguments:
        dataframe {Dataframe} -- Contiene palabras y sus n2v.
        palabra {Dataframe} -- Palabra a buscar dentro del dataframe

    Returns:
        int -- Retorna el índice si existe, sino retorna 0.

    TODO:
        Revisar por qué no está retornando correctamente los índices de lexicones,
        específicamente en náhuatl.

    """

    index = dataframe[dataframe[0] == palabra].index.get_values()
    if index.__len__():
        return int(index)
    return 0


def get_seed_index(lexicon_input, lexicon_target, source="seed"):
    """Esta función busca las palabras dentro de los dataframes y si existen
    obtiene los índices que ocupan dentro de los dataframes.

    Arguments:
        lexicon_input {Dataframe} -- Contiene vectores n2v de español.
        lexicon_target {Dataframe} -- Contiene vectores n2v de náhuatl.
        source {string} -- Lexicon a leer.

    Returns:
        list (2) -- Listas con índices de las palabras semillas.

    TODO:
        Agregar opción para leer set de evaluación o pruebas.
    """
    names = ["esp", "nah"]
    # Se lee el lexicon necesario
    if source.__eq__("seed"):
        lexiconsemilla = read_csv(
            "../lexiconessemilla/newlexiconb.lst", delimiter=" ", names=names)
    elif source.__eq__("eval"):
        lexiconsemilla = read_csv(
            "../lexiconesevaluacion/evaluationset", delimiter=" ", names=names)

    # print(lexiconsemilla.shape)
    semillas_esp = list(lexiconsemilla["esp"].values)
    semillas_nah = list(lexiconsemilla["nah"].values)

    pares = list(zip(semillas_esp, semillas_nah))

    # Busca vectores de las semillas, sino existen, lo descarta
    # lexicones en español.
    not_found = list()
    for i, palabra_es in enumerate(semillas_esp):
        if lexicon_input[lexicon_input[0] == palabra_es].shape[0] == 0:
            not_found.append(i)
    #print("es", not_found)
    # Índices de lexicones que no tienen vectores
    not_found = tuple(not_found)

    '''Muestra las palabras sin vectores
    for i in not_found:
        print(pares[i][0])
    '''
    pares = [v for i, v in enumerate(pares) if i not in frozenset(not_found)]
    '''
    with open("newlexicona.lst", "w") as fd:
        for _ in pares:
            fd.write(_[0] + " " + _[1] + "\n")
    '''
    ##################
    # Nuevo lexicon
    semillas_esp, semillas_nah = zip(*pares)
    # print(len(semillas_esp))
    del not_found
    # Busca vectores de las semillas, sino existen, lo descarta
    # lexicones en náhuatl.
    not_found = list()
    for i, palabra_na in enumerate(semillas_nah):
        if lexicon_target[lexicon_target[0] == palabra_na].shape[0] == 0:
            not_found.append(i)
    #print("nah", not_found)

    not_found = tuple(not_found)
    ''' Muestra palabras en náhuatl sin vectores
    for i in not_found:
        print(pares[i][1])'''
    pares = [v for i, v in enumerate(pares) if i not in frozenset(not_found)]
    # Genera un nuevo lexicon donde todas tienen sus correspondientes
    # representaciones vectoriales
    """
    with open("newlexiconb.lst", "w") as fd:
        for _ in pares:
            fd.write(_[0] + " " + _[1] + "\n")
    """
    semillas_esp, semillas_nah = zip(*pares)
    # print("asd",type(semillas_esp),len(semillas_nah))

    # Busca el índice del lexicon dentro de los dataframes para acceder a
    # sus vectores.
    index_esp = [int(lexicon_input[lexicon_input[0] == palabra].index[0])
                 for palabra in semillas_esp]

    index_nah = [int(lexicon_target[lexicon_target[0] == palabra].index[0])
                 for palabra in semillas_nah]

    return index_esp, index_nah


def get_vectors(dataframe, index, format=float64):
    """
    Retorna los vectores dentro del dataframe.

    Args:
        dataframe (Pandas.dataframe): Contiene las palabras y su representación vectorial.
        index (list): Contiene los índices que se necesitan del dataframe.
        format (numpy format): Tipo flotante. Default float64.

    Returns:
        Numpy array: Matriz con representaciones vectoriales.
    """

    return array([(dataframe.iloc[_].loc[1::])
                  for _ in index]).astype(format)


def make_hparam_string(*args):
    """Genera una cadena con los hiper parámetros para el LOGPATH

    Arguments:
        *args {str,int} -- hyperparámetros

    Returns:
        string -- Ruta del LOGPATH
    """

    return "./logs/NN_" + "".join([str(i) + "_" for i in args])


def get_top10_closest(vector, matrix, distance="cos"):
    """Calcular distancias entre vectores. La métrica por defecto es la distancia coseno.
    Se puede calcular la dist euclidiana.
    Se calcula la distancia de un vector a un arreglo de vectores. 

    Arguments:
        vector {numpy array} -- Vector a medir. 
        matrix {numpy array} -- Arreglo de vectores

    Keyword Arguments:
        distance {string} -- Argumento para especificar el tipo de métrica (default: {"cos"})

    Returns:
        {list} -- Regresa una lista de tuplas con el índice y las 10 distancias
                  más cercanas al vector.
    """

    # Distancia coseno
    if distance == "cos":
        # Medir distancias
        tmp_dist = list(enumerate(((matmul(vector, matrix.T)/(norm(vector)*sqrt(einsum('ij,ij->i',matrix, matrix)))))))
        # Se ordena lista según distancias más cercanas.
        tmp_dist = sorted(tmp_dist, key=lambda dist: dist[1], reverse=True)
        distances = tmp_dist[:10]
        del tmp_dist
        # Retorna 10 vectores más cercanos.
        return distances

    # Distancia euclidiana
    tmp_dist = [(i, norm(vector - matrix[i]))
                for i in range(matrix.shape[0])]
    tmp_dist = sorted(tmp_dist, key=lambda dist: dist[1])
    distances = tmp_dist[:10]
    del tmp_dist
    return distances


def get_closest_words_to(top_10, dataframe):
    """Muestra las 10 palabras más cercanas al vector generado por el modelo.

    Arguments:
        top_10 {list} -- Lista con índices y distancias más cercanas al vector 
                         generador por el modelo.
    """

    return [dataframe.iloc[index][0] for index, _ in top_10]


#import numpy as np


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists

    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if str(p) in actual and str(p) not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def next_batch(x, step, batch_size):
    """Función para obtener batches de un conjunto de datos

    Arguments:
        x {numpyarray} -- Conjunto de datos.
        step {int} -- Batches.
        batch_size {int} -- Tamaño del batch.

    Returns:
        numpyarray -- Subconjunto de tamaño batch_size.
    """

    return x[batch_size * step:batch_size * step + batch_size]
