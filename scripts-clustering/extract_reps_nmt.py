import os
from collections import defaultdict
import pickle
import sys

sys.path.append('.')

from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.embed_utils import extract_sent_reps_corpora, compute_doc_reps, read_doc_indexed_data


if __name__ == '__main__':
    BATCH_SIZE = 512
    LAYER_ID = 4

    model_name  = 'concat60'    
    hf_dir = f"experiments/en_et_{model_name}/hf"
    savedir = f"experiments/en_et_{model_name}/internals-docs"

    if not os.path.isdir(savedir):
        os.mkdir(savedir)
        
    tokenizer_hf = FSMTTokenizer.from_pretrained(hf_dir)
    model_hf = FSMTForConditionalGeneration.from_pretrained(hf_dir)
    model_hf = model_hf.cuda()
    encoder_hf = model_hf.base_model.encoder
    encoder_hf.device = model_hf.device

    # Sent embeddings
    encoded_sent = {}
    doc_ids = {}

    for split in ['dev-cl', 'test-cl', 'train']:
        print(split)
        data_dict_raw, doc_ids[split] = read_doc_indexed_data(split)

        encoded_sent[split] = extract_sent_reps_corpora(
            data_dict_raw, 
            tokenizer_hf, 
            encoder_hf, 
            layer_id=LAYER_ID, 
            batch_size=BATCH_SIZE
        )        

        savefile = f"{savedir}/sent_means_{split}.pkl"
        print(f"Saving to {savefile}")
        with open(savefile, 'wb') as f:
            pickle.dump(encoded_sent[split], f)
        
        print()

    # Doc embeddings
    encoded_doc = {}

    for k, v in encoded_sent.items():
        encoded_doc[k] = compute_doc_reps(encoded_sent[k], doc_ids[k])
    
    for split, v in encoded_doc.items():
        savefile = f"{savedir}/doc_encoded_{split}.pkl"
        print(f"Saving to {savefile}")
        with open(savefile, 'wb') as f:
            pickle.dump(v, f)
    
