from dynamicTimeWarping import *


def knn_predict(tr_d, te_d, dw, dw_lb, k_neighbors=5):
    distancemat, next_baskets = get_complete_distancematrix(te_d, tr_d, dw, dw_lb)

    all_pred = []
    all_dist = []

    for k in range(k_neighbors):
        knn_idx = distancemat.argsort()[:, :k]
        preds_k_l = []
        distances_k_l = []

        for i in range(len(te_d)):
            preds = [next_baskets[i][knn_idx[i][x]] for x in range(knn_idx.shape[1])]
            distances = np.mean([distancemat[i][knn_idx[i][x]] for x in range(knn_idx.shape[1])])
            pred_len = int(np.mean([len(te_d[i][x]) for x in range(len(te_d[i]))]))
            preds = [x for x, y in Counter([n for s in preds for n in s]).most_common(pred_len)]
            preds_k_l.append(preds)
            distances_k_l.append(distances)
        all_pred.append(preds_k_l)
        all_dist.append(distances_k_l)
    return all_pred, all_dist
