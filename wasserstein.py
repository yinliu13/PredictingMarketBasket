from scipy.stats import wasserstein_distance
import ot
from scipy.spatial.distance import cdist, pdist, squareform, euclidean
from product_embeddings import *
import numpy as np
import pandas as pd


def compute_lb(model, market_baskets):
    basket_X = model.wv.vectors[[x for x in market_baskets[0]]]
    basket_Y = model.wv.vectors[[x for x in market_baskets[1]]]

    # Compute first and second lower bound constraint using the Euclidean distance
    distance_matrix = cdist(basket_X, basket_Y)
    lb_1 = np.mean(np.min(distance_matrix, axis= 0))
    lb_2 = np.mean(np.min(distance_matrix, axis= 1))
    # take largest value of both lowerbounds
    lowerbound = max(lb_1, lb_2)
    return lowerbound


def get_dwasserstein(model, market_baskets):
    # make distance matrix d between purchase histories of customer c and d
    basket_X = market_baskets[0]
    basket_Y = market_baskets[1]
    list_basketX = list(basket_X)
    list_basketY = list(basket_Y)
    dictionary = np.unique(list_basketX +  list_basketY)
    dictionary_len = len(dictionary)
    product2index = dict(zip(dictionary, np.arange(dictionary_len)))

    dictionary_vectors = model.wv.vectors[[word for word in dictionary]]
    distance_matrix = squareform(pdist(dictionary_vectors))

    if np.sum(distance_matrix) == 0.0:
        return float('inf')

    def bag_of_words(document):
        bow = np.zeros(dictionary_len, dtype=np.float)
        for d in document:
            bow[product2index[d]] += 1.
        return bow / len(document)

    bow_X = bag_of_words(basket_X)
    bow_Y = bag_of_words(basket_Y)
    # Finally we compute the Wasserstein metric using both baskets and the distance metrics.
    dw = ot.emd2(bow_X, bow_Y, distance_matrix)
    return dw


