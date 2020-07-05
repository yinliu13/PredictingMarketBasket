import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from product_embeddings import*


def nested_embeddings(model, market_baskets):
    #removes all products without embeddings
    filtered_baskets = []
    for customer in market_baskets:
        newbasket = []
        for basket in customer:
            embedding = [product for product in basket if product in model.wv.vocab]
            len_embedding = len(embedding)
            if len_embedding > 0:
                newbasket.append(embedding)
        if len(newbasket) > 0:
            filtered_baskets.append(newbasket)
    return filtered_baskets


def get_purchase_history(all_baskets, l_b = 5, ph_lb_frequent = 10, ph_lb_occasional = 5):
    #l_b = minimum amount of products in each basket
    medium_baskets = []
    large_baskets = []
    for basket in all_baskets:
        purchased_baskets = []
        for items in basket:
            if len(items) > l_b:
                purchased_baskets.append(items)
        if len(purchased_baskets) > ph_lb_frequent:
            large_baskets.append(purchased_baskets)
        elif len(purchased_baskets) > ph_lb_occasional:
            medium_baskets.append(purchased_baskets)
    return medium_baskets, large_baskets


def double_check(product, str):
    # checks if products are still all strings, and if not, change all to string
    if isinstance(product, list):
        return [double_check(x, str) for x in product]
    return str(product)


def split_data(data):
    #split data into train, test and validation set
    train_fraction= 0.80
    validation_fraction = 0.10
    test_fraction = 0.10

    #First split data in train and test
    train, test = train_test_split(data, test_size=1 - train_fraction)
    #Next split the test data in test and validation data
    test, validation = train_test_split(test, test_size=test_fraction / (test_fraction + validation_fraction))

    test = [basket[:-1] for basket in test]  # test are all the baskets with which u predict
    test_target = [basket[-1] for basket in test]  # The last purchase of the customers that we aim to predict

    validation = [basket[:-1] for basket in validation]
    validation_target = [basket[-1] for basket in validation]

    return train, validation, validation_target, test, test_target


def get_frequent_products(market_baskets, amount=500):
    products = []
    for customer in market_baskets:
        for basket in customer:
            products.extend(basket)
    product_counter = Counter(products)
    frequent_products = [x for x, _ in product_counter.most_common(amount)]
    filtered_baskets = []
    for customer in market_baskets:
        filtered_customers = []
        for basket in customer:
            new_basket = [product for product in basket if product in frequent_products]
            if len(new_basket) > 0:
                filtered_customers.append(new_basket)
        if len(filtered_customers) > 0:
            filtered_baskets.append(filtered_customers)
    return filtered_baskets




class AisleBasketConstructor(object):
    '''
        Group products into baskets(type: list)
    '''

    def __init__(self, raw_data_dir, cache_dir):
        self.raw_data_dir = raw_data_dir
        self.cache_dir = cache_dir

    def get_orders(self):
        '''
            get order context information
        '''
        orders = pd.read_csv(self.raw_data_dir + 'orders.csv')
        orders = orders.fillna(0.0)
        orders['days'] = orders.groupby(['user_id'])['days_since_prior_order'].cumsum()
        orders['days_last'] = orders.groupby(['user_id'])['days'].transform(max)
        orders['days_up_to_last'] = orders['days_last'] - orders['days']
        del orders['days_last']
        del orders['days']
        return orders

    def get_orders_items(self, prior_or_train):
        '''
            get detailed information of prior or train orders
        '''
        orders_products = pd.read_csv(self.raw_data_dir + 'order_products__%s.csv' % prior_or_train)
        return orders_products

    def get_users_orders(self, prior_or_train):
        '''
            get users' prior detailed orders
        '''
        if os.path.exists(self.cache_dir + 'users_orders_aisle.pkl'):
            with open(self.cache_dir + 'users_orders_aisle.pkl', 'rb') as f:
                users_orders = pickle.load(f)
        else:
            orders = self.get_orders()
            order_products_prior = self.get_orders_items(prior_or_train)
            users_orders = pd.merge(order_products_prior,
                                    orders[['user_id', 'order_id', 'order_number', 'days_up_to_last']],
                                    on=['order_id'], how='left')
            with open(self.cache_dir + 'users_orders_aisle.pkl', 'wb') as f:
                pickle.dump(users_orders, f, pickle.HIGHEST_PROTOCOL)
        return users_orders

    def get_users_products(self, prior_or_train):
        '''
            get users' all purchased products
        '''
        if os.path.exists(self.cache_dir + 'users_aisle_products.pkl'):
            with open(self.cache_dir + 'users_aisle_products.pkl', 'rb') as f:
                users_products = pickle.load(f)
        else:
            users_products = self.get_users_orders(prior_or_train)[['user_id', 'aisle_id']].drop_duplicates()
            users_products['aisle_id'] = users_products.aisle_id.astype(int)
            users_products['user_id'] = users_products.user_id.astype(int)
            users_products = users_products.groupby(['user_id'])['aisle_id'].apply(list).reset_index()
            with open(self.cache_dir + 'users_aisle_products.pkl', 'wb') as f:
                pickle.dump(users_products, f, pickle.HIGHEST_PROTOCOL)
        return users_products

    def get_items(self, gran):
        '''
            get items' information
            gran = [departments, aisles, products]
        '''
        items = pd.read_csv(self.raw_data_dir + '%s.csv' % gran)
        return items

    def get_baskets(self, prior_or_train, reconstruct=False, none_idx=49689):
        '''
            get users' baskets
        '''
        filepath = self.cache_dir + './basket_' + prior_or_train + '.pkl'

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                up_basket = pickle.load(f)
        else:
            up = self.get_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'product_id'],
                                                                   ascending=True)
            uid_oid = up[['user_id', 'order_number']].drop_duplicates()
            up = up[['user_id', 'order_number', 'product_id']]
            up_basket = up.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index()
            up_basket = pd.merge(uid_oid, up_basket, on=['user_id', 'order_number'], how='left')
            for row in up_basket.loc[up_basket.product_id.isnull(), 'product_id'].index:
                up_basket.at[row, 'product_id'] = [none_idx]
            up_basket = up_basket.sort_values(['user_id', 'order_number'], ascending=True).groupby(['user_id'])[
                'product_id'].apply(list).reset_index()
            up_basket.columns = ['user_id', 'basket']
            with open(filepath, 'wb') as f:
                pickle.dump(up_basket, f, pickle.HIGHEST_PROTOCOL)
        return up_basket

    def get_aisle_baskets(self, prior_or_train, reconstruct=False, none_idx=49689):
        '''
                    get users' baskets
                '''
        filepath = self.cache_dir + './basket_aisle' + prior_or_train + '.pkl'

        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                up_basket = pickle.load(f)
        else:
            up = self.get_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'aisle_id'],
                                                                   ascending=True)
            uid_oid = up[['user_id', 'order_number']].drop_duplicates()
            up = up[['user_id', 'order_number', 'aisle_id']]
            up_basket = up.groupby(['user_id', 'order_number'])['aisle_id'].apply(list).reset_index()
            up_basket = pd.merge(uid_oid, up_basket, on=['user_id', 'order_number'], how='left')
            for row in up_basket.loc[up_basket.aisle_id.isnull(), 'aisle_id'].index:
                up_basket.at[row, 'aisle_id'] = [none_idx]
            up_basket = up_basket.sort_values(['user_id', 'order_number'], ascending=True).groupby(['user_id'])[
                'aisle_id'].apply(list).reset_index()
            up_basket.columns = ['user_id', 'basket']
            with open(filepath, 'wb') as f:
                pickle.dump(up_basket, f, pickle.HIGHEST_PROTOCOL)
        return up_basket

    def get_item_history(self, prior_or_train, reconstruct=False, none_idx=49689):
        filepath = self.cache_dir + './item_history_' + prior_or_train + '.pkl'
        if (not reconstruct) and os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                item_history = pickle.load(f)
        else:
            up = self.get_users_orders(prior_or_train).sort_values(['user_id', 'order_number', 'product_id'],
                                                                   ascending=True)
            item_history = up.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index()
            item_history.loc[item_history.order_number == 1, 'product_id'] = item_history.loc[
                                                                                 item_history.order_number == 1, 'product_id'] + [
                                                                                 none_idx]
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending=True)
            # accumulate
            item_history['product_id'] = item_history.groupby(['user_id'])['product_id'].transform(pd.Series.cumsum)
            # get unique item list
            item_history['product_id'] = item_history['product_id'].apply(set).apply(list)
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending=True)
            # shift each group to make it history
            item_history['product_id'] = item_history.groupby(['user_id'])['product_id'].shift(1)
            for row in item_history.loc[item_history.product_id.isnull(), 'product_id'].index:
                item_history.at[row, 'product_id'] = [none_idx]
            item_history = item_history.sort_values(['user_id', 'order_number'], ascending=True).groupby(['user_id'])[
                'product_id'].apply(list).reset_index()
            item_history.columns = ['user_id', 'history_items']

            with open(filepath, 'wb') as f:
                pickle.dump(item_history, f, pickle.HIGHEST_PROTOCOL)
        return item_history

class TaFengBasketConstructor(object):
    def __init__(self):
        pass

    def get_baskets(self):
        path = "./data/user_tran.csv"
        data = pd.read_csv(path)

        df = pd.DataFrame()
        df['User Id'] = data['CUSTOMER_ID']
        df['Transaction Id'] = data['TRANSACTION_ID']
        df['Product Id'] = data['PRODUCT_ID']

        userid = df['User Id'].values
        unique_users = np.unique(userid)
        transactionids = data['TRANSACTION_ID'].values
        unique_transactionids = np.unique(transactionids)

        all_baskets = []
        for user in range(len(unique_users)):
            all_baskets.append([])
            #print(unique_users[user])
            df_tmp = df[df['User Id'] == unique_users[user]]
            #print(df_tmp)
            unique = np.unique(df_tmp['Transaction Id'])
            #print(unique)
            for trans in unique:
                all_baskets[-1].append(list(df_tmp[df_tmp['Transaction Id'] == trans]['Product Id'].values))
        #print(all_baskets)

        return all_baskets