import os
from collections import defaultdict
import pickle
import sys

sys.path.append('.')

from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.embed_utils import extract_sent_reps_corpora, compute_doc_reps


def extract_sent_reps_all_domains(split, savedir):
    # split in 'test', 'valid', 'train'
    
    data_dict_raw = defaultdict(list)
    doc_ids = defaultdict(list)

    domain_names = ["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
    for domain_name in domain_names:
        fn = f"experiments/doc-indices/sp-cl-{domain_name}.en-et.docs.{split}.both"
        with open(fn) as f:
            for l in f.readlines():
                doc_ids[domain_name].append(l[:-1].split('\t')[0])
                data_dict_raw[domain_name].append(l[:-1].split('\t')[1])
    
    BATCH_SIZE = 512
    LAYER_ID = 4
    
    data_dict_encoded = extract_sent_reps_corpora(
        data_dict_raw, 
        tokenizer_hf, 
        encoder_hf, 
        layer_id=LAYER_ID, 
        batch_size=BATCH_SIZE
    )
    
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    savefile = f"{savedir}/sent_means_{split}.pkl"
    print(f"Saving to {savefile}")
    with open(savefile, 'wb') as f:
        pickle.dump(data_dict_encoded, f)
    
    return data_dict_encoded, doc_ids


if __name__ == '__main__':
        
    model_name  = 'concat60'    
    hf_dir = f"experiments/en_et_{model_name}/hf"
    savedir = f"experiments/en_et_{model_name}/internals-docs"

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
        encoded_sent[split], doc_ids[split] = extract_sent_reps_all_domains(split, savedir)
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
    
