import pandas as pd
from gensim.models import Word2Vec
import multiprocessing


def get_embeddings_product():
    # load data
    traindata = pd.read_csv('./data/order_products__train.csv')
    priordata = pd.read_csv('./data/order_products__prior.csv')

    # turn productIDs into strings first to feed to Word2Vec
    traindata["product_id"] = traindata["product_id"].astype(str)
    priordata["product_id"] = priordata["product_id"].astype(str)

    products_train = traindata.groupby("order_id")
    products_train_order = products_train.apply(lambda x: list(x["product_id"]))
    products_prior = priordata.groupby("order_id")
    products_prior_order = products_prior.apply(lambda x: list(x['product_id']))

    print("this is train", products_train_order)
    print("First of train data", products_train_order[1])

    all_orders = products_prior_order.append(products_train_order).values

    # Train embeddings
    cores = multiprocessing.cpu_count()
    print('All orders of Instacart- P', all_orders)
    w2v = Word2Vec(all_orders, size=50, window=5, min_count=50, workers=cores - 1)
    w2v.save('product_instacart_embeddings.bin')



def get_embeddings_aisle():
    #load data
    traindata = "./data/order_products__train.csv"
    priordata = "./data/order_products__prior.csv"
    products = "./data/products.csv"

    df_train = pd.read_csv(traindata)
    df_prior = pd.read_csv(priordata)
    products = pd.read_csv(products)
    #print(len(df_prior.order_id))

    # Assign the right category to each product, for the prior data as well as the train data
    aisles_prior = []
    for product in df_prior.product_id:
        number = products[products['product_id'] == product].aisle_id.values.astype(str)
        aisles_prior.append(number[0])
    df_prior['aisle_id'] = aisles_prior
    print(df_prior)
    print(df_prior)

    aisles_train = []
    for product in df_train.product_id:
        print(product)
        number = products[products['product_id'] == product].aisle_id.values.astype(str)
        aisles_train.append(number[0])
    df_train['aisle_id'] = aisles_train

    print('ordering')
    # Merge the two datasets
    train_products = df_train.groupby("order_id").apply(lambda x: list(x['aisle_id']))
    prior_products = df_prior.groupby("order_id").apply(lambda x: list(x['aisle_id']))

    all_orders = prior_products.append(train_products).values

    # Train embeddings
    cores = multiprocessing.cpu_count()
    print('Word2Vec model')
    model = Word2Vec(all_orders, size=50, window=5, min_count=50, workers=cores - 1)
    model.save("aisles_embeddings.model")
    model.wv.save_word2vec_format("aisles_embeddings.model", binary=True)


def get_embeddings_tafeng():
    # load data, this data does not need to be merged
    data = pd.read_csv("./data/user_tran.csv")

    data["PRODUCT_ID"] = data["PRODUCT_ID"].astype(str)
    presentence = data.groupby("TRANSACTION_ID").apply(lambda x: list(x['PRODUCT_ID']))
    print(presentence)
    finalorders = presentence.values
    print(finalorders)

    # Train embeddings
    cores = multiprocessing.cpu_count()
    model = Word2Vec(finalorders, size=50, window=5, min_count=50, workers=cores - 1)
    model.save("tafeng_embeddings.model")
    model.wv.save_word2vec_format("tafeng_embeddings.bin", binary=True)


if __name__ == "__main__":
    get_embeddings_tafeng()