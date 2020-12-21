import sys

sys.path.append('.')

from da.clust_utils import train_kmeans_doc_sent

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
    train_kmeans_doc_sent(savedir)
