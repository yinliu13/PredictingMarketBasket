import numpy as np
import itertools

def get_top_nc(all_baskets, iterations):
    #calculates the average amount of products ordered for each customer
    top_nc_all = []
    for c in range(iterations):

        sizes = []
        for b in range(len(all_baskets[c])):
            sizes.append(len(all_baskets[c][b]))
        top_nc = sum(sizes) / len(sizes)
        top_nc_all.append(top_nc)
    return top_nc_all


def get_cartesian(all_baskets, uniqueitems):
    # calculates the cartesian products of all products
    all_association = []
    all_cartesian = []
    matrix = np.zeros((len(uniqueitems), len(uniqueitems)))
    for customer in range(len(all_baskets)):
        print('Customer', customer)
        for basket in range(
                len(all_baskets[customer]) - 1):  # compute all cartesian products for each consecutive basket
            cartesian = list(
                itertools.product(all_baskets[customer][basket], all_baskets[customer][basket + 1]))
            all_cartesian = all_cartesian + cartesian
        # print(all_cartesian)
        # total_cartesian = cartesianproduct + cartesian2
        # rtesian)
        # for cp in total_cartesian:
        #   print(cp)
        # all_cartesian = np.unique(all_cartesian)
    #construct matrix
    for tuple in all_cartesian:
        # print(tuple)
        # print(tuple[0])
        # print(np.where(uniqueitems == tuple[0]))
        # print(np.where(uniqueitems == tuple[1]))
        rowcount = np.where(uniqueitems == tuple[0])
        colcount = np.where(uniqueitems == tuple[1])
        # print(rowcount[-1])
        matrix[rowcount[-1], colcount[-1]] += 1
        # print('Count matrix', matrix)
    for customer in range(len(all_baskets)):
        prediction = np.array([])
        basket_m = all_baskets[customer][-1]
        # print('Most recent basket', basket_m)
        for product in range(len(basket_m)):
            index = np.where(uniqueitems == basket_m[product])
            try:
                pred_product = np.max(matrix[index[-1], :])
                finalindex = np.where(matrix[index[-1], :] == pred_product)

                # print('predicted product', pred_product)
                # print(finalindex)
                # print('product', uniqueitems[finalindex[-1]])
                prediction = np.append(prediction, uniqueitems[finalindex[-1]])
            except ValueError:
                pass
        all_association.append(list(prediction))
    return all_association
