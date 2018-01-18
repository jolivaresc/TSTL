import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy.random import uniform




x = (uniform(0,15,size=[100,128]))
y = (uniform(0,15,size=[100,128]))


def get_distance(vector, matrix,distance="cos"):
    
    if distance == "cos":
        tmp_dist = [(i,1 - ((dot(vector,matrix[i]))/(norm(vector)*norm(matrix[i]))))
                     for i in range(matrix.shape[0])]
        tmp_dist = sorted(tmp_dist, key = lambda dist: dist[1])
        distances = tmp_dist[:10]
        del tmp_dist
        return distances
    
    
    tmp_dist = [(i,norm(vector-matrix[i]))
                 for i in range(matrix.shape[0])]
    tmp_dist = sorted(tmp_dist, key= lambda dist: dist[1])
    distances = tmp_dist[:10]
    del tmp_dist
    return distances





dist = get_distance(x[0],x)

print(dist)

dist.sort(key=lambda tup: tup[1])

print("AAAAAAAa")
print(dist)

