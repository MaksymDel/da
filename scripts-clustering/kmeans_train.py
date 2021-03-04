import sys

sys.path.append('.')

from da.clust_utils import train_kmeans_sent, train_kmeans_doc

if __name__ == '__main__':
    exp = sys.argv[1]
    langpair = sys.argv[2]
    src_lang, tgt_lang = langpair.split("-")

    if exp == 'nmt':
        model_name  = 'concat60'

    elif exp == 'bert':
        model_name  = 'xlm-roberta-base'
        
    else:
        raise ValueError("Wrong argument")
    
    savedir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/internals-docs"
    
    NUM_CLUSTERS=3
    #train_kmeans_sent(savedir, NUM_CLUSTERS)
    train_kmeans_doc(savedir, NUM_CLUSTERS)
