from wasserstein import *
import wasserstein
import numpy as np

from collections import Counter

def dynamic_time_warping_distance(seq_c, seq_d, shortestsofar, dw, d_lb): #make distance matrix d between purchase histories of customer c and d
    m = len(seq_d)
    n = len(seq_c)

    for i in range(int(n)):
        for j in range(int(m)):
            lowerbound = wasserstein.compute_lb(seq_c[i], seq_d[j])

    lowerbound_min = np.fromiter(lowerbound, dtype=np.float)
    #If this clearly not the shortest option, breakpoint
    if np.sum(lowerbound_min[np.argpartition(lowerbound_min, m)][m]) > max(shortestsofar):
        return np.inf, seq_d[0]

    dtw_matrix = np.inf * np.ones((m , n))

    # Compute all distances
    distancematrix = np.zeros((m, n))

    for i in range(n):
        for j in range(m):
            distancematrix[i, j] = dw((seq_c[i], seq_d[j]))

        # assign values to first row and column of matrix
        dtw_matrix[0, 0] = dw((seq_c[0], seq_d[0]))
        for i in range(1, m):
            dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + distancematrix[i, 0]

        for j in range(1, n):
            dtw_matrix[0, j] = distancematrix[0, j]

            # calculate other values according to recurrent forumla
            for i in range(1, m):
                w = 1.
                for j in range(1, n):
                    choices = dtw_matrix[i - 1, j - 1], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j]
                    dtw_matrix[i, j] = min(choices) + w * distancematrix[i, j]

            min_idx = np.argmin(dtw_matrix[-1, :-1])
            d_dtw = dtw_matrix[-1, min_idx]
            prediction_next_basket =seq_d[min_idx + 1]
            return d_dtw , prediction_next_basket


def get_complete_distancematrix(c, d, dw, dw_lowerbound, tau = 10, k = 5):
    c_shape = np.shape(c)
    d_shape = np.shape(d)
    distancemat = np.inf* np.ones(c_shape[0], d_shape[0])
    next_baskets = np.empty(c_shape[0], d_shape[0])

    for i in (range(c_shape[0])):
        c[i] = np.array(c[i])
        if c[i].shape[0] > tau: # if it exceeds tau, revert to fallback prediction.
            c[i] = c[i][-tau]
            best_dist = [np.inf] * max(k)
        for j in range(d_shape[0]):
            d[i] = np.array(d[i])
            dist, pred = dynamic_time_warping_distance(c[i], d[j], best_dist, dw, dw_lowerbound)
            if dist < np.max(best_dist):
                best_dist[np.argmax(best_dist)] = dist
            distancemat[i, j] = dist
            next_baskets[i,j] = pred
        return distancemat, next_baskets

