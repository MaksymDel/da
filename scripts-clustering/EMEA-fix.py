import os
import sys

sys.path.append('.')

from transformers import AutoModel, AutoTokenizer
from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.embed_utils import extract_reps_doc, extract_reps_sent


# STEP 1.1: extract sent reps

savedir = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/EMEA-fix"
split="test-cl"

filename_data_doc_indexed = f"/gpfs/hpc/projects/nlpgroup/bergamot/backup/multidomain/data-raw/cl-EMEA.en-et.docs.{split}"
filename_savefile_sentmeans = f"{savedir}/sent_means_{split}.pkl"

model_name  = 'xlm-roberta-base'
tokenizer_hf = AutoTokenizer.from_pretrained(model_name)
encoder_hf = AutoModel.from_pretrained(model_name).cuda()
BATCH_SIZE = 1500 # probably can do 512
LAYER_ID = 7

extract_reps_sent(
    filename_data_doc_indexed=filename_data_doc_indexed,
    filename_savefile=filename_savefile_sentmeans,
    tokenizer_hf=tokenizer_hf,
    encoder_hf=encoder_hf,
    batch_size=BATCH_SIZE,
    layer_id=LAYER_ID
    )

# STEP 1.2: extract doc reps
filename_loadfile_sent_means = filename_savefile_sentmeans
filename_savefile_docmeans = f"{savedir}/doc_means_{split}.pkl"
filename_savefile_docids = f"{savedir}/docids_{split}.pkl"

extract_reps_doc(
    filename_data_doc_indexed=filename_data_doc_indexed,
    filename_sent_means=filename_loadfile_sent_means,
    filename_savefile_doc_means=filename_savefile_docmeans,
    filename_savefile_doc_ids=filename_savefile_docids
    )


# # STEP 2: apply clustering model to get clusters
# kmeans_model_file = "/gpfs/hpc/projects/nlpgroup/bergamot/backup/multidomain/experiments-multidomain/en_et_xlm-roberta-base/internals-docs/kmeans_train_doc.pkl"

# with open(kmeans_model_file, 'rb') as f:
#     kmeans = pickle.load(f)

# data_file = f"{internals_dir}/doc_encoded_{split}.pkl"

# sp-cl-Europarl.de-en.docs.test.both

# # STEP 3: compare to original EMEA clusters as a sanity check; if match, repeat for not dedub set