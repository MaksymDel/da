# embed not dedub EMEA test set

# savedir
savedir = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/EMEA-fix"

# doc-indices file
# data_file_wrong = "/gpfs/hpc/projects/nlpgroup/bergamot/backup/multidomain/experiments-multidomain/doc-indices/sp-cl-EMEA.en-et.docs.test-cl.both"
data_file = "/gpfs/hpc/projects/nlpgroup/bergamot/backup/multidomain/data-raw/cl-EMEA.de-en.docs.test-cl"

# clustering model
kmeans_model_file = "/gpfs/hpc/projects/nlpgroup/bergamot/backup/multidomain/experiments-multidomain/en_et_xlm-roberta-base/internals-docs/kmeans_train_doc.pkl"

# STEP 1.1: extract sent reps
langpair = "en-et"
model_name  = 'xlm-roberta-base'
BATCH_SIZE = 3000
LAYER_ID = 4

src_lang, tgt_lang = langpair.split("-")

model_name  = 'concat60'    
hf_dir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/hf"
savedir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/internals-docs"
os.makedirs(savedir, exist_ok=True)

tokenizer_hf = FSMTTokenizer.from_pretrained(hf_dir)
model_hf = FSMTForConditionalGeneration.from_pretrained(hf_dir)
model_hf = model_hf.cuda()
encoder_hf = model_hf.base_model.encoder
encoder_hf.device = model_hf.device

# STEP 1.2: extract doc reps


# STEP 2: apply clustering model to get clusters
with open(kmeans_model_file, 'rb') as f:
    kmeans = pickle.load(f)

data_file = f"{internals_dir}/doc_encoded_{split}.pkl"

sp-cl-Europarl.de-en.docs.test.both

# STEP 3: compare to original EMEA clusters as a sanity check; if match, repeat for not dedub set