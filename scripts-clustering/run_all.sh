# assumes you trained fairseq models e.g. with `scripts/fairseq`

# NMT
python scripts/convert_chkp_fseq_to_hf.py
python scripts/extract_reps_nmt.py
python scripts/train_kmeans_nmt.py

# BERT
python scripts/extract_reps_bert.py
python scripts/train_kmeans_bert.py

# now you can run notebooks from `notebooks` folder 

