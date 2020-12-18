import sys

sys.path.append('.')

from da.clust_utils import train_kmeans_doc_sent

if __name__ == '__main__':
    exp = sys.argv[1]

    if exp == 'nmt':
        model_name  = 'concat60'
        savedir = f"experiments/en_et_{model_name}/internals-docs"
        train_kmeans_doc_sent(savedir)

    elif exp == 'bert':
        model_name  = 'xlm-roberta-base'
        savedir = f"experiments/en_et_{model_name}/internals-docs"
        train_kmeans_doc_sent(savedir)

    else:
        raise ValueError("Wrong argument")