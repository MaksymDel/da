# srun --mem=190G -t 64:00:00 --pty bash
# module load python/3.6.3
# conda activate da2
# 
# da:  python scripts-clustering/kmeans_train.py nmt de-en sent 12
# da-doc: python scripts-clustering/kmeans_train.py nmt de-en doc 12
# da3: python scripts-clustering/kmeans_train.py bert de-en sent 12
# da4: python scripts-clustering/kmeans_train.py bert de-en doc 12

# 
# da:  python scripts-clustering/kmeans_predict.py nmt de-en sent 12
# da-doc: python scripts-clustering/kmeans_predict.py nmt de-en doc 12
# da3: python scripts-clustering/kmeans_predict.py bert de-en sent 12
# da4: python scripts-clustering/kmeans_predict.py bert de-en doc 12

# fairseq-generate data-bin/ --path chkp/checkpoint60.pt --sacrebleu --quiet --gen-subset valid


# GPU:
# srun -p gpu --gres gpu:tesla:1 --mem=30G -t 192:00:00 --pty bash
# srun --mem=70G -t 192:00:00 --pty bash 

# python scripts-clustering/kmeans_train.py bert sent 8
# # python scripts-clustering/kmeans_train.py nmt sent 8
