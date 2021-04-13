import os
import pickle
import numpy as np
import scipy.stats as ss
from sklearn.cluster import KMeans

from .embed_utils import read_doc_indexed_data, pickle_load_from_file, pickle_dump_to_file


def predict_uniq_clustids(kmeans_model_file, filename_docmeans, filename_savefile_uniq_clustids):

    kmeans_model = pickle_load_from_file(kmeans_model_file)
    uniq_docmeans = pickle_load_from_file(filename_docmeans)
    
    
    uniq_docmeans = np.array(uniq_docmeans, dtype=np.float32)

    uniq_clustids = kmeans_model.predict(uniq_docmeans)

    pickle_dump_to_file(uniq_clustids, filename_savefile_uniq_clustids)
    
    return uniq_clustids

def docids_to_clustlabels(filename_doc_indixed, filename_uniq_docids, filename_uniq_clustids, filename_savefile_clastids_hat):
    _, docids_for_labeling = read_doc_indexed_data(filename_doc_indixed)
    uniq_docids = pickle_load_from_file(filename_uniq_docids)
    uniq_clustids = pickle_load_from_file(filename_uniq_clustids)
    
    id2clust = dict(zip(uniq_docids, uniq_clustids))

    clustids_hat = [str(id2clust[id]) for id in docids_for_labeling] 

    print(f"Saving to {filename_savefile_clastids_hat}")
    with open(f"{filename_savefile_clastids_hat}", "w") as f:
        for clustid in clustids_hat:
            f.write(f"{clustid}\n")
    print("Saved")

    return clustids_hat

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
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10, verbose=55).fit(all_encoded) # 5 for original results
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





def train_kmeans_doc(savedir, NUM_CLUSTERS):
    np.random.seed(21)

    # Train k-means on document embeddings
    doc_enc_path = f"{savedir}/doc_means_train.pkl"

    with open(doc_enc_path, 'rb') as f:
        data_encoded = pickle.load(f)
        

    kmeans_docs = kmeans_train(data_encoded, NUM_CLUSTERS)

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    fn = f"{savedir}/new_kmeans_train_doc_{NUM_CLUSTERS}.pkl"
    print(f"Saving to {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(kmeans_docs, f)


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
    print(kcorr, rcorr, phi2corr)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


