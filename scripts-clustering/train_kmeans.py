import sys

sys.path.append('.')

from da.clust_utils import train_kmeans_doc_sent

if __name__ == '__main__':
    # NMT
    model_name  = 'concat60'
    savedir = f"experiments/en_et_{model_name}/internals-docs"

    train_kmeans_doc_sent(savedir)

    # BERT
    model_name  = 'xlm-roberta-base'
    savedir = f"experiments/en_et_{model_name}/internals-docs"

    train_kmeans_doc_sent(savedir)