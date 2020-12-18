import os
import sys

sys.path.append('.')

from transformers import AutoModel, AutoTokenizer
from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.embed_utils import extract_reps_doc_sent


def configure_nmt():
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

    return {'savedir': savedir, 'tokenizer_hf': tokenizer_hf, 'encoder_hf': encoder_hf, 'layer_id': LAYER_ID, 'batch_size': BATCH_SIZE} 


def configure_bert():
    BATCH_SIZE = 256 # probably can do 512
    LAYER_ID = -2 # corresponds to layer 11

    model_name  = 'xlm-roberta-base'    
    savedir = f"experiments/en_et_{model_name}/internals-docs"

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    model_hf = AutoModel.from_pretrained(model_name)
    tokenizer_hf = AutoTokenizer.from_pretrained(model_name)

    model_hf = model_hf.cuda()
    encoder_hf = model_hf

    return {'savedir': savedir, 'tokenizer_hf': tokenizer_hf, 'encoder_hf': encoder_hf, 'layer_id': LAYER_ID, 'batch_size': BATCH_SIZE} 


if __name__ == '__main__':
    exp = sys.argv[1]

    if exp == 'nmt':
        nmt_args = configure_nmt() 
        extract_reps_doc_sent(**nmt_args)

    elif exp == 'bert':
        bert_args = configure_bert()
        extract_reps_doc_sent(**bert_args)

    else:
        raise ValueError("Wrong argument")