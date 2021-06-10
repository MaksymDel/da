import pickle
from collections import defaultdict
import torch
import numpy as np



def read_doc_indexed_data(filename_data_doc_indexed):
    print(f"Loading from {filename_data_doc_indexed}")

    data_raw = []
    doc_ids = []

    with open(filename_data_doc_indexed, "r") as f:
        for l in f.readlines():
            if len(l.rstrip().split('\t')) >= 2:
                doc_ids.append(l.rstrip().split('\t')[0])
                data_raw.append(l.rstrip().split('\t')[1])
            else:
                data_raw.append(l.rstrip().split('\t')[0])


    print("Loaded")
    return data_raw, doc_ids


def extract_reps_sent_batch(src, tokenizer_hf, encoder_hf, layer_id):
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


def extract_reps_sent(
                        data,
                        tokenizer_hf, 
                        encoder_hf, 
                        batch_size, 
                        layer_id 
                        ):
 
    # Sent embeddings
    encoded_sent = []

    print("Extracting reps...")
    it = 0
    for i in range(0, len(data), batch_size):
        if it % 10 == 0:
            print(it)


        batch = data[i:i+batch_size]
        encoded_sent.extend(extract_reps_sent_batch(batch, tokenizer_hf, encoder_hf, layer_id))
        it += 1

    return encoded_sent


def compute_doc_reps(all_encoded, all_ids):

    ids_to_reps = defaultdict(list)
    for id, rep in zip(all_ids, all_encoded):
        ids_to_reps[id].append(rep)
    
    del all_encoded

    for k, v in ids_to_reps.items():
        ids_to_reps[k] = np.array(v).mean(0)
    
    return list(ids_to_reps.values()), list(ids_to_reps.keys())


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


def pickle_dump_to_file(obj, fn):
    print(f"Saving to {fn}")
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
    print("Saved")


def pickle_load_from_file(fn):
    print(f"Loading from {fn}")
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    print("Loaded")
    return obj

# def compute_doc_reps_old(data_encoded, doc_ids):

#     all_encoded = []
#     all_ids = []

#     for d, v in data_encoded.items():
#         all_encoded.extend(data_encoded[d])
#         all_ids.extend(doc_ids[d])

#     #all_encoded = np.array(all_encoded)
#     #all_ids = np.array(all_ids)
#     #print('b')
    
#     ids_to_reps = defaultdict(list)
#     for id, rep in zip(all_ids, all_encoded):
#         ids_to_reps[id].append(rep)
    
#     del all_encoded

#     for k, v in ids_to_reps.items():
#         ids_to_reps[k] = np.array(v).mean(0)
    
#     doc_embedded_corpus = []
#     for id in all_ids:
#         doc_embedded_corpus.append(ids_to_reps[id])

#     res_dict = {}    
#     i = 0
#     for d, v in data_encoded.items():
#         res_dict[d] = doc_embedded_corpus[i:i+len(v)]
#         i += len(v)
#     return res_dict