import numpy as np
import scipy.stats as ss
from sklearn.cluster import KMeans

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

    all_encoded, labels_true = flatten_dict(data_encoded)  

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=5).fit(all_encoded)

    return kmeans


def kmeans_predict(kmeans_model, data_encoded, train=False):

    all_encoded, labels_true = flatten_dict(data_encoded)  
    
    if train:
        labels_hat = kmeans_model.labels_
    else:
        labels_hat = kmeans_model.predict(all_encoded)
    
    return labels_hat, labels_true