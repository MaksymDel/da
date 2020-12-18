# assumes you trained fairseq models e.g. with `scripts/fairseq`
python scripts/convert_chkp_fseq_to_hf.py
python scripts/extract_reps_nmt.py
python scripts/train_kmeans_nmt.py
# now you can run notebooks from `notebooks` folder 