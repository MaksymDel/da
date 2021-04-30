import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append('.')

from da.clust_utils import predict_kmeans_sent, predict_kmeans_doc

np.random.seed(21)


if __name__ == '__main__':
    encoder_type = sys.argv[1]
    sent_or_doc = sys.argv[2]
    NUM_CLUSTERS = int(sys.argv[3])
    split = sys.argv[4] # train / dev / test

    assert encoder_type in ["nmt", "bert"]
    assert sent_or_doc in ["sent", "doc"]
    assert split in ["train", "dev", "test", "dev-cl", "test-cl"]
    
    # CHANGE THIS LINE
    exp_folder = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/en_cs_ParaCrawl"
    data_exp_folder = "/gpfs/hpc/projects/nlpgroup/bergamot/paracrawl-cz-en/"
    #

    savedir = f"{exp_folder}/outputs/{encoder_type}"
    savedir_clusters = f"{exp_folder}/outputs/cluster_labels/"
    os.makedirs(savedir_clusters, exist_ok=True)
    
    filename_kmeans_model = f"{savedir}/kmeans_{encoder_type}_{sent_or_doc}_{NUM_CLUSTERS}.pkl"
    filename_embeddings = f"{savedir}/{sent_or_doc}_means_{split}.pkl"

    filename_savefile_labels = f"{savedir_clusters}/ParaCrawl.{split}.{encoder_type}.{sent_or_doc}.k_{NUM_CLUSTERS}.clustlabels"

    if sent_or_doc == "sent":
        predict_kmeans_sent(filename_kmeans_model, filename_embeddings, filename_savefile_labels)
    elif sent_or_doc == "doc":
        filename_uniq_docids = f"{savedir}/docids_{split}.pkl"
        filename_doc_indixed_to_label = f"{data_exp_folder}/cl-ParaCrawl.en-cs.docs.{split}" # only for docids

        predict_kmeans_doc(filename_kmeans_model, filename_embeddings, filename_uniq_docids, filename_doc_indixed_to_label, filename_savefile_labels)
    else:
        raise ValueError("should be 'sent' or 'doc'")