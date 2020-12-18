import os
import pickle
import sys
import numpy as np

sys.path.append('.')

from da.clust_utils import kmeans_train


if __name__ == '__main__':
    np.random.seed(21)

    model_name  = 'concat60'
    savedir = f"experiments/en_et_{model_name}/internals-docs"

    # Train k-means on sentence embeddings
    sent_enc_path = f"{savedir}/sent_means_train.pkl"

    with open(sent_enc_path, 'rb') as f:
        data_encoded = pickle.load(f)
        
    NUM_CLUSTERS=4

    kmeans_sent = kmeans_train(data_encoded, NUM_CLUSTERS)

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    fn = f"{savedir}/kmeans_train_sent.pkl"
    print(f"Saving to {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(kmeans_sent, f)


    # Train k-means on document embeddings
    doc_enc_path = f"{savedir}/doc_encoded_train.pkl"

    with open(doc_enc_path, 'rb') as f:
        data_encoded = pickle.load(f)
        
    NUM_CLUSTERS=4

    kmeans_docs = kmeans_train(data_encoded, NUM_CLUSTERS)

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    fn = f"{savedir}/kmeans_train_doc.pkl"
    print(f"Saving to {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(kmeans_docs, f)
