from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def kmeans_encode(kmeans, data, nb_clusters):
    enc_data = []
    i = 0
    for example in data:
        example_data = []
        Z = kmeans.predict(example)
        for (index, point) in enumerate(example):
            if (point[0] == -1 and point[1] == 1):
                example_data.append(nb_clusters)
            elif (point[0] == -1 and point[1] == -1):
                example_data.append(nb_clusters + 1)
            else:
                example_data.append(Z[index])
        enc_data.append(example_data)
        i += 1
    return enc_data