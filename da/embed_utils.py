import pickle
from collections import defaultdict
import torch
import numpy as np


def extract_sent_reps(src, tokenizer_hf, encoder_hf, layer_id):
    # tok
    src = tokenizer_hf.batch_encode_plus(
        src,
        padding="longest", 
        return_tensors="pt",
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        max_length=100
    )
    # res
    for k, v in src.items():
        src[k] = v.to(encoder_hf.device)
    
    with torch.no_grad():
        res = encoder_hf.forward(**src,
                           return_dict=True,
                           output_hidden_states=True,
                           #output_attentions=True,
                           )
    
    #he = [r.detach().cpu().numpy() for r in res['hidden_states']]
    
    he = res['hidden_states'][layer_id]
    
    he_means = masked_mean(he, src['attention_mask'].unsqueeze(2).bool(), 1)
        
    return he_means.detach().cpu().numpy()


def extract_sent_reps_corpora(data_dict_raw, tokenizer_hf, encoder_hf, layer_id, batch_size):

    data_dict_encoded = defaultdict(list)

    for domain, data in data_dict_raw.items():
        print(f"Encoding {domain} data...")

        it = 0
        for i in range(0, len(data), batch_size):
            if it % 100 == 0:
                print(it)

            batch = data[i:i+batch_size]
            data_dict_encoded[domain].extend(extract_sent_reps(batch, tokenizer_hf, encoder_hf, layer_id))

            it += 1

    return data_dict_encoded


def compute_doc_reps(data_encoded, doc_ids):

    all_encoded = []
    all_ids = []

    for d, v in data_encoded.items():
        all_encoded.extend(data_encoded[d])
        all_ids.extend(doc_ids[d])
    print('a')

    #all_encoded = np.array(all_encoded)
    #all_ids = np.array(all_ids)
    #print('b')
    
    ids_to_reps = defaultdict(list)
    for id, rep in zip(all_ids, all_encoded):
        ids_to_reps[id].append(rep)
    print('c')
    
    del all_encoded

    for k, v in ids_to_reps.items():
        ids_to_reps[k] = np.array(v).mean(0)
    print('d')
    
    doc_embedded_corpus = []
    for id in all_ids:
        doc_embedded_corpus.append(ids_to_reps[id])
    print('e')

    res_dict = {}    
    i = 0
    for d, v in data_encoded.items():
        res_dict[d] = doc_embedded_corpus[i:i+len(v)]
        i += len(v)
    print('f')
    return res_dict


def read_doc_indexed_data(
                        split, 
                        langpair,
                        exp,
                        domain_names=["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
                        ):
    src_lang, tgt_lang = langpair.split("-")

    data_dict_raw = defaultdict(list)
    doc_ids = defaultdict(list)

    for domain_name in domain_names:
        if exp == "nmt": # with sp
            fn = f"experiments/doc-indices/sp-cl-{domain_name}.{src_lang}-{tgt_lang}.docs.{split}.both"
        elif exp == "bert":
            fn = f"experiments/doc-indices/cl-{domain_name}.{src_lang}-{tgt_lang}.docs.{split}"
        else:
            raise ValueError(f"wrong domain name: {domain_name}")
        
        print(f"Loading {fn}")
        with open(fn) as f:
            for l in f.readlines():
                doc_ids[domain_name].append(l[:-1].split('\t')[0])
                data_dict_raw[domain_name].append(l[:-1].split('\t')[1])

    return data_dict_raw, doc_ids



def extract_reps_doc_given_sent(
                        savedir,tokenizer_hf, 
                        encoder_hf, 
                        batch_size, 
                        layer_id, 
                        langpair,
                        exp,
                        domain_names=["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
                        ):
    # Sent embeddings
    encoded_sent = {}
    doc_ids = {}

    for split in ['dev', 'test', 'train']:
        print(split)
        _, doc_ids[split] = read_doc_indexed_data(split, langpair, exp, domain_names)
        del _

        savefile = f"{savedir}/sent_means_{split}.pkl"
        print(f"Loading from {savefile}")
        with open(savefile, 'rb') as f:
            encoded_sent[split] = pickle.load(f)
        
        print("Loaded")
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



def extract_reps_doc_sent(
                        savedir,tokenizer_hf, 
                        encoder_hf, 
                        batch_size, 
                        layer_id, 
                        langpair,
                        exp,
                        domain_names=["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"],
                        ):
    # Sent embeddings
    encoded_sent = {}
    doc_ids = {}

    for split in ['dev', 'test', 'train']:
        print(split)
        data_dict_raw, doc_ids[split] = read_doc_indexed_data(split, langpair, exp, domain_names)

        encoded_sent[split] = extract_sent_reps_corpora(
            data_dict_raw, 
            tokenizer_hf, 
            encoder_hf, 
            layer_id=layer_id, 
            batch_size=batch_size
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


def masked_mean(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """
    # taken from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

    To calculate mean along certain dimensions on masked values
    # Parameters
    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension
    # Returns
    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    def tiny_value_of_dtype(dtype: torch.dtype):
        if not dtype.is_floating_point:
            raise TypeError("Only supports floating point dtypes.")
        if dtype == torch.float or dtype == torch.double:
            return 1e-13
        elif dtype == torch.half:
            return 1e-4
        else:
            raise TypeError("Does not support dtype " + str(dtype))

    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))