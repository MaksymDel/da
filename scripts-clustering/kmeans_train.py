import sys

sys.path.append('.')

from da.clust_utils import train_kmeans

if __name__ == '__main__':
    encoder_type = sys.argv[1]
    sent_or_doc = sys.argv[2]
    NUM_CLUSTERS = int(sys.argv[3])

    assert encoder_type in ["nmt", "bert"]
    assert sent_or_doc in ["sent", "doc"]
    # assert split in ["train", "dev", "test", "dev-cl", "test-cl"]

    # CHANGE THIS LINE
    basedir = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/en_cs_ParaCrawl"
    #
    
    savedir = f"{basedir}/outputs/{encoder_type}"
    
    filename_embeddings = f"{savedir}/{sent_or_doc}_means_train.pkl"
    filename_savefile_kmeans_model = f"{savedir}/kmeans_{encoder_type}_{sent_or_doc}_{NUM_CLUSTERS}.pkl"

    train_kmeans(NUM_CLUSTERS, filename_embeddings, filename_savefile_kmeans_model)
    