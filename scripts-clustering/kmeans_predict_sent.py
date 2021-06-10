import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append('.')

from da.clust_utils import predict_kmeans_sent, predict_kmeans_doc

np.random.seed(21)


if __name__ == '__main__':
    filename_kmeans_model = sys.argv[1]
    filename_embeddings = sys.argv[2]
    filename_savefile_labels = sys.argv[3]

    predict_kmeans_sent(filename_kmeans_model, filename_embeddings, filename_savefile_labels)
