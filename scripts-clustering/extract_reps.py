import os
import sys

sys.path.append('.')

from transformers import AutoModel, AutoTokenizer
from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.embed_utils import extract_reps_doc, extract_reps_sent


def configure_nmt_doc(langpair, exp_type='multidomain'):
    print("Configuring NMT")
    src_lang, tgt_lang = langpair.split("-")

    model_name  = 'concat60'    
    hf_dir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/hf"
    savedir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/internals-docs"
    os.makedirs(savedir, exist_ok=True)

    if exp_type == 'multidomain':
        domain_names = ["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
    elif exp_type == 'paracrawl':
        domain_names = ["base"]
        # domain_names = ["ParaCrawl"]
    else:
        raise ValueError(f"{exp_type} is not correct")
    
    _ = ""

    return {
            'savedir': savedir, 
            'tokenizer_hf': _, 
            'encoder_hf': _, 
            'layer_id': _, 
            'batch_size': _, 
            'langpair': langpair,
            'exp': 'nmt',
            'domain_names': domain_names
            } 


def configure_nmt_sent(langpair, exp_type='multidomain'):
    print("Configuring NMT")
    src_lang, tgt_lang = langpair.split("-")

    BATCH_SIZE = 3000
    LAYER_ID = 4

    model_name  = 'concat60'    
    hf_dir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/hf"
    savedir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/internals-docs"
    os.makedirs(savedir, exist_ok=True)

    tokenizer_hf = FSMTTokenizer.from_pretrained(hf_dir)
    model_hf = FSMTForConditionalGeneration.from_pretrained(hf_dir)
    model_hf = model_hf.cuda()
    encoder_hf = model_hf.base_model.encoder
    encoder_hf.device = model_hf.device

    if exp_type == 'multidomain':
        domain_names = ["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
    elif exp_type == 'paracrawl':
        domain_names = ["ParaCrawl"]
    else:
        raise ValueError(f"{exp_type} is not correct")
        
    return {
            'savedir': savedir, 
            'tokenizer_hf': tokenizer_hf, 
            'encoder_hf': encoder_hf, 
            'layer_id': LAYER_ID, 
            'batch_size': BATCH_SIZE, 
            'langpair': langpair,
            'exp': 'nmt',
            'domain_names': domain_names
            } 


def configure_bert_sent(langpair, exp_type='multidomain'):
    print("Configuring BERT")
    src_lang, tgt_lang = langpair.split("-")

    BATCH_SIZE = 1500 # probably can do 512
    LAYER_ID = 7

    model_name  = 'xlm-roberta-base'    
    savedir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/internals-docs"
    os.makedirs(savedir, exist_ok=True)

    model_hf = AutoModel.from_pretrained(model_name)
    tokenizer_hf = AutoTokenizer.from_pretrained(model_name)
    model_hf = model_hf.cuda()
    encoder_hf = model_hf

    if exp_type == 'multidomain':
        domain_names = ["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
    elif exp_type == 'paracrawl':
        domain_names = ["ParaCrawl"]
    else:
        raise ValueError(f"{exp_type} is not correct")
        
    return {
            'savedir': savedir, 
            'tokenizer_hf': tokenizer_hf, 
            'encoder_hf': encoder_hf, 
            'layer_id': LAYER_ID, 
            'batch_size': BATCH_SIZE, 
            'langpair': langpair,
            'exp': 'bert',
            'domain_names': domain_names
            }


def configure_bert_doc(langpair, exp_type='multidomain'):
    print("Configuring BERT")
    src_lang, tgt_lang = langpair.split("-")

    model_name  = 'xlm-roberta-base'    
    savedir = f"experiments/{src_lang}_{tgt_lang}_{model_name}/internals-docs"
    os.makedirs(savedir, exist_ok=True)

    if exp_type == 'multidomain':
        domain_names = ["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
    elif exp_type == 'paracrawl':
        domain_names = ["ParaCrawl"]
    else:
        raise ValueError(f"{exp_type} is not correct")
    
    _ = ""

    return {
            'savedir': savedir, 
            'tokenizer_hf': _, 
            'encoder_hf': _, 
            'layer_id': _, 
            'batch_size': _, 
            'langpair': langpair,
            'exp': 'bert',
            'domain_names': domain_names
            }


if __name__ == '__main__':
    exp = sys.argv[1] # nmt / bert
    langpair = sys.argv[2] # en-et / de-en
    exp_type = sys.argv[3] # multidomain / paracrawl
    sent_or_doc = sys.argv[4] # sent / doc

    if sent_or_doc == "sent":
        if exp == 'nmt':
            args = configure_nmt_sent(langpair, exp_type)
        elif exp == 'bert':
            args = configure_bert_sent(langpair, exp_type)

        extract_reps_sent(**args)

    elif sent_or_doc == 'doc':
        if exp == 'nmt':
            args = configure_nmt_doc(langpair, exp_type)
        elif exp == 'bert':
            args = configure_bert_doc(langpair, exp_type)

        extract_reps_doc(**args)
    
    else:
        raise ValueError("Wrong argument")

    #extract_reps_doc_sent(**args)
    #extract_reps_doc_given_sent(**args)