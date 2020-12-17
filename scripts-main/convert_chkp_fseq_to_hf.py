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


def chkp60():
    # chkp 60
    exp_path = f"experiments/en_et_concat60"
    fseq_checkpoint_path = f"{exp_path}/checkpoint60.pt"
    save_dir = f"{exp_path}/hf"
    data_path = f"experiments/bin-data-en-et-base"
    spm_model_file=None

    convert_fsmt_checkpoint_to_pytorch(
        fseq_checkpoint_path, 
        save_dir, 
        data_path, 
        spm_model_file
        )

    # test
    test(save_dir)


def chkp101():
    # chkp 101
    exp_path = f"experiments/en_et_concat101"
    fseq_checkpoint_path = f"{exp_path}/checkpoint101.pt"
    save_dir = f"{exp_path}/hf"
    data_path = f"experiments/bin-data-en-et-base"
    spm_model_file=None

    convert_fsmt_checkpoint_to_pytorch(
        fseq_checkpoint_path, 
        save_dir, 
        data_path, 
        spm_model_file
        )

    test(save_dir)


def chkp1():
    # chkp 1
    exp_path = f"experiments/en_et_concat1"
    fseq_checkpoint_path = f"{exp_path}/checkpoint1.pt"
    save_dir = f"{exp_path}/hf"
    data_path = f"experiments/bin-data-en-et-base"
    spm_model_file=None

    convert_fsmt_checkpoint_to_pytorch(
        fseq_checkpoint_path, 
        save_dir, 
        data_path, 
        spm_model_file
        )

    test(save_dir)


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
        
        convert_fsmt_checkpoint_to_pytorch(
            fseq_checkpoint_path, 
            save_dir, 
            data_path, 
            spm_model_file
            )

        test(save_dir)


if __name__ == '__main__':
    chkp60()