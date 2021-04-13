import numpy as np
import logging
from argparse import ArgumentParser


sys.path.append('.')

split="test"
split_predict="test-cl"
emea_dir = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/EMEA-fix"

test_raw = f"/gpfs/hpc/projects/nlpgroup/bergamot/backup/multidomain/data-raw/cl-EMEA.en-et.docs.{split_predict}"
test_clusters = f"{emea_dir}/cl-EMEA.en-et.docs.{split}.clustids"

def split_by_clusters



python scripts-clustering/separate_clusters.py --path_to_files experiments/EMEA-fix/concat-docs-data/ \
--input_files sp-cl-EMEA.en-et.docs.test-cl \
--src_lang en --tgt_lang et --n_clusters 4 \
--indices cl-EMEA.en-et.docs.test.clustids \
--cluster_mode_name bert_clusters_doc_emea