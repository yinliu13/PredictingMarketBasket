
y_true = [[1,2,3] , [1, 2, 5], [4, 3, 8], [2, 3, 4]]
y_pred = [[1,2,3], [1, 2, 4], [2, 1], [2, 4, 3]]



def jaccard_coefficient(y_pred, y_true):
    count = []
    for basket in range(len(y_pred)):
        common_i = 0
        merge = set(y_pred[basket]).union(set(y_true[basket]))
        #print(merge)
        intersection = set(y_pred[basket]).intersection(set(y_true[basket]))
        #print('products in common', len(intersection))
        jaccard = len(intersection)/len(merge)
        count.append(jaccard)
    print('Jaccard', count)
    jacc_coef = sum(count)/len(count)
    return jacc_coef


def my_f1_score(y_pred, y_true):
    # initialize values
    tp = 0
    fp = 0
    fn = 0
    all_f1scores = []

    for basket in range(len(y_pred)):
        # print('predicted basket is ', y_pred[basket])
        true_positives = set(y_pred[basket]).intersection(set(y_true[basket]))
        #print('predict in basket, actually in basket', true_positives)
        tp = tp + len(true_positives)
        false_positives = set(y_pred[basket]).difference(set(y_true[basket]))
        #print('predict in basket, NOT actually', false_positives)
        fp = fp + len(false_positives)
        false_negatives = set(y_true[basket]).difference(set(y_pred[basket]))
        #print('predict NOT in basket but ACTUALLY in basket', list(false_negatives))
        fn = fn + len(false_negatives)
        f1score = 2 * tp / (2 * tp + fp + fn)
        #print(f1score)
        all_f1scores.append(f1score)
    print(all_f1scores)
    average_f1 = sum(all_f1scores) / len(all_f1scores)
    print(average_f1)
    return average_f1





