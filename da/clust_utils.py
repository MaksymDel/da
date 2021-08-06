import os
import pickle
import numpy as np
import scipy.stats as ss
from sklearn.cluster import KMeans

from .embed_utils import read_doc_indexed_data, pickle_load_from_file, pickle_dump_to_file


def label_embeddings(kmeans_model_file, filename_embeddings):
    kmeans_model = pickle_load_from_file(kmeans_model_file)
    embeddings = pickle_load_from_file(filename_embeddings)
    # embeddings = np.array(embeddings, dtype=np.float32)
    labels = kmeans_model.predict(embeddings)

    return labels    


def predict_kmeans_sent(filename_kmeans_model, filename_embeddings, filename_savefile_labels):
    final_labels = label_embeddings(filename_kmeans_model, filename_embeddings)
    print(f"Saving cluster labels to {filename_savefile_labels}")
    np.savetxt(f"{filename_savefile_labels}", final_labels.astype(int), fmt="%i")


def predict_kmeans_doc(filename_kmeans_model, filename_embeddings, filename_uniq_docids, filename_doc_indixed_to_label, filename_savefile_labels):
    uniq_labels = label_embeddings(filename_kmeans_model, filename_embeddings)
    uniq_docids = pickle_load_from_file(filename_uniq_docids)
    docid2label = dict(zip(uniq_docids, uniq_labels))

    _, docids_for_labeling = read_doc_indexed_data(filename_doc_indixed_to_label)
    final_labels = [docid2label[id] for id in docids_for_labeling]
    
    print(f"Saving cluster labels to {filename_savefile_labels}")
    final_labels = np.array(final_labels)
    np.savetxt(f"{filename_savefile_labels}", final_labels.astype(int), fmt="%i")

    
# def predict_uniq_clustids(kmeans_model_file, filename_docmeans, filename_savefile_uniq_clustids):

#     kmeans_model = pickle_load_from_file(kmeans_model_file)
#     uniq_docmeans = pickle_load_from_file(filename_docmeans)
    
#     uniq_docmeans = np.array(uniq_docmeans, dtype=np.float32)

#     uniq_clustids = kmeans_model.predict(uniq_docmeans)

#     pickle_dump_to_file(uniq_clustids, filename_savefile_uniq_clustids)
    
#     return uniq_clustids

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

# def flatten_dict(dct):
#     all_encoded = []
#     all_labels = []

#     for d in dct.keys():
#         all_encoded.extend(dct[d])
#         all_labels.extend([d] * len(dct[d]))   
    

#     all_encoded = np.array(all_encoded)
#     all_labels = np.array(all_labels)

#     return all_encoded, all_labels
    

# def kmeans_predict(kmeans_model, data_encoded, train=False):

#     all_encoded, labels_true = flatten_dict(data_encoded)  
    
#     if train:
#         labels_hat = kmeans_model.labels_
#     else:
#         labels_hat = kmeans_model.predict(all_encoded)
    
#     return labels_hat, labels_true


def train_kmeans(NUM_CLUSTERS, filename_embeddings, filename_savefile_kmeans_model):
    np.random.seed(21)
    data_encoded = pickle_load_from_file(filename_embeddings)
    kmeans_model = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init=10, verbose=55).fit(data_encoded) # 5 for original results
    pickle_dump_to_file(kmeans_model, filename_savefile_kmeans_model)


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


