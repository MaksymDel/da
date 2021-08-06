import sys

sys.path.append('.')

#from transformers import FSMTForConditionalGeneration
from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.fsmt.convert_fsmt_original_pytorch_checkpoint_to_pytorch import convert_fsmt_checkpoint_to_pytorch
from types import MethodType
from da.greedy_search_interpret import greedy_search_interpret


def test(save_dir):
    tokenizer_hf = FSMTTokenizer.from_pretrained(save_dir) #, spm_model=f"{save_dir}/spm_model.spm")
    model_hf = FSMTForConditionalGeneration.from_pretrained(save_dir)
    
    # Monkeypatch the function on object
    model_hf.greedy_search = MethodType(greedy_search_interpret, model_hf)

    src = "▁So ▁be ▁it ."
    sentence = tokenizer_hf.encode_plus(
        src,
        padding="longest", 
        return_tensors="pt",
        return_token_type_ids=False,
        return_attention_mask=False
    )

    # tokenizer_hf.batch_decode(sentence['input_ids'])

    res = model_hf.generate(**sentence,
                       #return_dict=True,
                       output_hidden_states=True,
                       output_attentions=True,
                       do_sample=False,
                       num_beams=1)


    assert len(res['encoder_hidden_states']) == 7
    assert list(res['encoder_hidden_states'][0].shape) == [1,5,512]


# def chkp60_para(langpair):
#     src_lang, tgt_lang = langpair.split("-")
#     # chkp 60
#     exp_path = f"experiments/{src_lang}_{tgt_lang}_paracrawl"
#     fseq_checkpoint_path = f"{exp_path}/checkpoint60.pt"
#     save_dir = f"{exp_path}/hf"
#     data_path = f"{exp_path}/data-fseq-bin"
#     spm_model_file=None

#     return {"fsmt_checkpoint_path": fseq_checkpoint_path,
#             "pytorch_dump_folder_path": save_dir, 
#             "data_path": data_path, 
#             "spm_model_path": spm_model_file}

def encs_paracrawl_3m():
    basedir_load = "/gpfs/hpc/projects/nlpgroup/bergamot/paracrawl-cz-en/data"
    basedir_save = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/en_cs_ParaCrawl" 
    fseq_checkpoint_path = f"{basedir_save}/mt_chkp/checkpoint60.pt"
    data_path = f"{basedir_load}/bin-data-en-cs-ParaCrawl"
    save_dir = f"{basedir_save}/hf"
    spm_model_file=None


    return {"fsmt_checkpoint_path": fseq_checkpoint_path,
            "pytorch_dump_folder_path": save_dir, 
            "data_path": data_path, 
            "spm_model_path": spm_model_file}

def deen_paracrawl_3m_20m_params():
    basedir_load = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/de_en_ParaCrawl_3m/data"
    basedir_save = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/de_en_ParaCrawl_3m_20m_params" 
    fseq_checkpoint_path = f"{basedir_load}/models/de_en_ParaCrawl_3m_20m_params/checkpoint60.pt"
    data_path = f"{basedir_load}/bin-data-de-en-ParaCrawl"
    save_dir = f"{basedir_save}/hf"
    spm_model_file=None


    return {"fsmt_checkpoint_path": fseq_checkpoint_path,
            "pytorch_dump_folder_path": save_dir, 
            "data_path": data_path, 
            "spm_model_path": spm_model_file}



def deen_paracrawl_3m():
    basedir = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/de_en_ParaCrawl_3m"
    fseq_checkpoint_path = f"{basedir}/data/models/de_en_ParaCrawl_3m/checkpoint60.pt"
    save_dir = f"{basedir}/hf"
    data_path = f"{basedir}/data/bin-data-de-en-ParaCrawl"
    spm_model_file=None


    return {"fsmt_checkpoint_path": fseq_checkpoint_path,
            "pytorch_dump_folder_path": save_dir, 
            "data_path": data_path, 
            "spm_model_path": spm_model_file}


def chkp60(langpair):
    src_lang, tgt_lang = langpair.split("-")
    # chkp 60
    exp_path = f"experiments/{src_lang}_{tgt_lang}_concat60"
    fseq_checkpoint_path = f"{exp_path}/checkpoint60.pt"
    save_dir = f"{exp_path}/hf"
    data_path = f"experiments/bin-data-{src_lang}-{tgt_lang}-base"
    spm_model_file=None

    return {"fsmt_checkpoint_path": fseq_checkpoint_path,
            "pytorch_dump_folder_path": save_dir, 
            "data_path": data_path, 
            "spm_model_path": spm_model_file}


def chkp101():
    # chkp 101
    exp_path = f"experiments/en_et_concat101"
    fseq_checkpoint_path = f"{exp_path}/checkpoint101.pt"
    save_dir = f"{exp_path}/hf"
    data_path = f"experiments/bin-data-en-et-base"
    spm_model_file=None

    return {"fsmt_checkpoint_path": fseq_checkpoint_path,
            "pytorch_dump_folder_path": save_dir, 
            "data_path": data_path, 
            "spm_model_path": spm_model_file}


def chkp1():
    # chkp 1
    exp_path = f"experiments/en_et_concat1"
    fseq_checkpoint_path = f"{exp_path}/checkpoint1.pt"
    save_dir = f"{exp_path}/hf"
    data_path = f"experiments/bin-data-en-et-base"
    spm_model_file=None

    return {"fsmt_checkpoint_path": fseq_checkpoint_path,
            "pytorch_dump_folder_path": save_dir, 
            "data_path": data_path, 
            "spm_model_path": spm_model_file}


def chkp_tuned():
    # Convert checkpoint
    domain_names = ["Europarl", "OpenSubtitles", "JRC-Acquis", "EMEA"]
    for main_name in domain_names:
        # Fine-tuned models
        exp_path = f"experiments/en_et_{main_name}_ft"
        fseq_checkpoint_path = f"{exp_path}/checkpoint100.pt"
        save_dir = f"{exp_path}/hf"
        data_path = f"experiments/bin-data-en-et-{main_name}-ft/"
        spm_model_file=None
        
        return {"fsmt_checkpoint_path": fseq_checkpoint_path,
                "pytorch_dump_folder_path": save_dir, 
                "data_path": data_path, 
                "spm_model_path": spm_model_file}


if __name__ == '__main__':
    langpair = sys.argv[1] # de-en
    
    #chkp60()
    #args = chkp60(langpair)
    #args = chkp60_para(langpair)
    
    # CHANGE THIS LINE
    #args = deen_paracrawl_3m_20m_params()
    args = encs_paracrawl_3m()
    #

    convert_fsmt_checkpoint_to_pytorch(**args)

    test(args['pytorch_dump_folder_path'])