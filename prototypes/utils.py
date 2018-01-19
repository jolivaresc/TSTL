
# coding: utf-8

from numpy import dot, float64, array
from numpy.linalg import norm
from pandas import set_option, read_csv

__author__ = "Olivares Castillo José Luis"


# Se establece la precisión con la que Pandas lee el archivo para evitar que
# trunque el valor de los vectores
set_option('display.max_colwidth', -1)
set_option('precision', 18)


def load_node2vec():
    """Esta función lee los archivos para almacenar los vectores node2vec del español
    y náhuatl los retorna en dataframes de Pandas.
    
    Returns:
        Dataframe (2) -- dataframes con palabras - node2vec.
    """

    es = read_csv("../vectors/es.node2vec.embeddings",
                  delimiter=" ", skiprows=1, header=None)
    nah = read_csv("../vectors/na.node2vec.embeddings",
                   delimiter=" ", skiprows=1, header=None)

    return es, nah


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


def get_seed_index(lexicon_input, lexicon_target, source ="training_set"):
    """Esta función obtiene los índices de las palabras semillas de los
    dataframes.
    
    Arguments:
        lexicon_input {Dataframe} -- Contiene vectores n2v de español.
        lexicon_target {Dataframe} -- Contiene vectores n2v de náhuatl.
        source {string} -- Indica si se lee el lexicon semilla o de evaluación.
    
    Returns:
        list (2) -- Listas con índices de las palabras semillas.
    
    TODO:
        Agregar opción para leer set de evaluación o pruebas.
    """
    names = ["esp", "nah"]
    # Se lee el lexicon necesario
    lexiconsemilla = read_csv("../lexiconessemilla/lexicon.esna.proc.norep.tmp2",delimiter=" ", names=names)


    # Se almacenan las palabras semillas de español y náhuatl en listas.
    semillas_esp = list(lexiconsemilla["esp"].values)
    semillas_nah = list(lexiconsemilla["nah"].values)

    # Se buscan los índices de las palabras semilla en los dataframes para obtener sus
    # representaciones vectoriales.
    # Nota: Se omite la palabra semilla si no existe su representación vectorial.
    index_esp = [int(lexicon_input[lexicon_input[0] == palabra].index.get_values())
                 for palabra in semillas_esp
                 if int(lexicon_input[lexicon_input[0] == palabra].index.get_values().__len__()) == 1]
    index_nah = [int(lexicon_target[lexicon_target[0] == palabra].index.get_values())
                 for palabra in semillas_nah
                 if int(lexicon_target[lexicon_target[0] == palabra].index.get_values().__len__()) == 1]

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

    return "./logs/NN_" + "".join([str(i)+"_" for i in args])


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
    #return 1 - (np.matmul(vector, matrix.T) / (np.linalg.norm(vector) *
    #  np.sqrt(np.einsum('ij,ij->i', matrix, matrix))))
    
    # [np.linalg.norm(x[_]) for _ in range(x.shape[0])] 
    # => np.sqrt((x*x).sum(axis=1))
    if distance == "cos":
        tmp_dist = [(i, 1 - ((dot(vector, matrix[i])) / (norm(vector) * norm(matrix[i]))))
                    for i in range(matrix.shape[0])]
        tmp_dist = sorted(tmp_dist, key=lambda dist: dist[1])
        distances = tmp_dist[:10]
        del tmp_dist
        return distances

    # Distancia euclidiana
    tmp_dist = [(i, norm(vector - matrix[i]))
                for i in range(matrix.shape[0])]
    tmp_dist = sorted(tmp_dist, key=lambda dist: dist[1])
    distances = tmp_dist[:10]
    del tmp_dist
    return distances


def get_closest_words_to(top_10,dataframe):
    """Muestra las 10 palabras más cercanas al vector generado por el modelo.
    
    Arguments:
        top_10 {list} -- Lista con índices y distancias más cercanas al vector 
                         generador por el modelo.
    """ 

    for index, _ in top_10:
        print(dataframe.iloc[index][0])
