

import os
import pickle
import numpy as np
import scipy.stats as ss
from sklearn.cluster import KMeans
from numpy import array, array_equal, allclose


def flatten_dict(dct):
    all_encoded = []
    all_labels = []

    for d in dct.keys():
        all_encoded.extend(dct[d])
        all_labels.extend([d] * len(dct[d]))   
    

    all_encoded = np.array(all_encoded)
    all_labels = np.array(all_labels)

    return all_encoded, all_labels
    

def kmeans_train(data_encoded, num_clusters):

    print("start flattenning")
    all_encoded, _ = flatten_dict(data_encoded)  
    print("start clustering")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10, verbose=20).fit(all_encoded)
    print("finished")
    return kmeans


def kmeans_predict(kmeans_model, data_encoded, train=False):

    all_encoded, labels_true = flatten_dict(data_encoded)  
    
    if train:
        labels_hat = kmeans_model.labels_
    else:
        labels_hat = kmeans_model.predict(all_encoded)
    
    return labels_hat, labels_true


def train_kmeans_sent(savedir, NUM_CLUSTERS):
    np.random.seed(21)

    # Train k-means on sentence embeddings
    sent_enc_path = f"{savedir}/sent_means_train.pkl"

    print(f"Loading from {sent_enc_path}")
    with open(sent_enc_path, 'rb') as f:
        data_encoded = pickle.load(f)
        

    kmeans_sent = kmeans_train(data_encoded, NUM_CLUSTERS)

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    fn = f"{savedir}/kmeans_train_sent_{NUM_CLUSTERS}.pkl"
    print(f"Saving to {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(kmeans_sent, f)


def train_kmeans_doc_weighted(savedir, NUM_CLUSTERS):
    np.random.seed(21)

    doc_enc_path = f"{savedir}/doc_encoded_train.pkl"

    with open(doc_enc_path, 'rb') as f:
        data_encoded = pickle.load(f)
        

    kmeans_docs = kmeans_train(data_encoded, NUM_CLUSTERS)

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    fn = f"{savedir}/kmeans_train_doc_{NUM_CLUSTERS}.pkl"
    print(f"Saving to {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(kmeans_docs, f)


# def train_kmeans_doc(savedir, NUM_CLUSTERS):
#     np.random.seed(21)

#     # Train k-means on document embeddings
#     doc_enc_path = f"{savedir}/doc_encoded_train.pkl"

#     with open(doc_enc_path, 'rb') as f:
#         data_encoded = pickle.load(f)

#     #test for exact equality
#     def arreq_in_list(myarr, list_arrays):
#         return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

#     def merge_duplicates(list_of_arrs):
#         res = []
#         for arr in list_of_arrs:
#             if arreq_in_list(arr, res):
#                 pass
#             else:
#                 res.append(arr)
#         return res

#     print("Merging duplicates")
#     # merge duplicates
#     for k, v in data_encoded.items():
#         data_encoded[k] = merge_duplicates(v)


#     kmeans_docs = kmeans_train(data_encoded, NUM_CLUSTERS)

#     if not os.path.isdir(savedir):
#         os.mkdir(savedir)

#     fn = f"{savedir}/kmeans_train_doc_{NUM_CLUSTERS}.pkl"
#     print(f"Saving to {fn}")
#     with open(fn, 'wb') as f:
#         pickle.dump(kmeans_docs, f)


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


