import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.append('.')

from da.clust_utils import cramers_corrected_stat, kmeans_predict

np.random.seed(21)

# NMT sent clusters
def predict_class_labels(exp, langpair, NUM_CLUSTERS):

    if exp == 'nmt':
        model_name  = 'concat60'

    elif exp == 'bert':
        model_name  = 'xlm-roberta-base'
        
    else:
        raise ValueError("{exp} is a wrong argument")

    #for clustering_type in ["sent", "doc"]:
    for clustering_type in ["doc"]:

        print(clustering_type)

        src_lang, tgt_lang = langpair.split('-') 

        model_dir = f"experiments/{src_lang}_{tgt_lang}_{model_name}"
        internals_dir = f"{model_dir}/internals-docs"
        clusters_dir = f"{model_dir}/clustering_data"

        if clustering_type == "sent":
            kmeans_model_file = f"{internals_dir}/kmeans_train_sent_{NUM_CLUSTERS}.pkl"
            savedir = f"{clusters_dir}/{NUM_CLUSTERS}/{exp}-clusters-sent"

        elif clustering_type == "doc":
            kmeans_model_file = f"{internals_dir}/kmeans_train_doc_{NUM_CLUSTERS}.pkl"
            savedir = f"{clusters_dir}/{NUM_CLUSTERS}/{exp}-clusters-doc"

        else:
            raise ValueError(f"{clustering_type} is not known")

        with open(kmeans_model_file, 'rb') as f:
            kmeans = pickle.load(f)
        
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        for split in ['train', 'dev', 'test']:
            print(split)
            print("###")
            
            if clustering_type == "sent":
                data_file = f"{internals_dir}/sent_means_{split}.pkl"
            elif clustering_type == "doc":
                data_file = f"{internals_dir}/doc_encoded_{split}.pkl"
            else:
                raise ValueError(f"{clustering_type} is not known")

            with open(data_file, 'rb') as f:
                data_encoded = pickle.load(f)
            
            labels_hat, labels_true = kmeans_predict(kmeans, data_encoded, split=='train')
            
            conf_matrix = pd.crosstab(labels_true, labels_hat)
            corr_k = cramers_corrected_stat(conf_matrix)

            print(f"Corr k: {corr_k}")
            print(conf_matrix)

            print(f"Saving to {savedir}")
            i = 0
            for domain_name, v in data_encoded.items():

                if clustering_type == "sent":
                    fn = f"{domain_name}.{split}.clust.{exp}.sent"
                elif clustering_type == "doc":
                    fn = f"{domain_name}.{split}.clust.{exp}.doc"
                else:
                    raise ValueError(f"{clustering_type} is not known")
                
                np.savetxt(f"{savedir}/{fn}", labels_hat[i:i+len(v)].astype(int), fmt="%i")
                i += len(v)

            # sns.heatmap(conf_matrix,
            #             cmap="YlGnBu", annot=True, cbar=False)
            # plt.show()
            
            print()


if __name__ == '__main__':
    exp = sys.argv[1]
    langpair = sys.argv[2]
    src_lang, tgt_lang = langpair.split("-")
    
    NUM_CLUSTERS=4
    predict_class_labels(exp, langpair, NUM_CLUSTERS)
