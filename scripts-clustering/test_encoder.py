
import sys
import torch
from types import MethodType

sys.path.append('.')

from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.greedy_search_interpret import greedy_search_interpret

if __name__ == '__main__':
        
    main_name = "concat60"
    hf_dir = f"experiments/en_et_{main_name}/hf"

    tokenizer_hf = FSMTTokenizer.from_pretrained(hf_dir)
    model_hf = FSMTForConditionalGeneration.from_pretrained(hf_dir)
    model_hf = model_hf.cuda()
    model_hf.greedy_search = MethodType(greedy_search_interpret, model_hf)

    data = ["▁how ▁would ▁you ▁like ▁that ▁done ▁to ▁you ?", 
    "▁- ▁if ▁you ▁have ▁ever ▁had ▁chemotherapy ▁with ▁a ▁drug ▁called ▁doxorubicin ▁or ▁a ▁drug ▁related ▁to", 
    "▁it ▁is ▁one ▁of ▁the ▁most ▁difficult ▁jobs ▁in ▁the ▁European ▁Union ▁that ▁he ▁has ▁to ▁do ."] 

    data_tok = tokenizer_hf.batch_encode_plus(
            data,
            padding="longest", 
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True
        )

    for k, v in data_tok.items():
        data_tok[k] = v.to(model_hf.device)

    res_generate = model_hf.generate(**data_tok,
                    #return_dict=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    do_sample=False,
                    num_beams=1)

    res_forward = model_hf.forward(**data_tok,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=True)

    encoder = model_hf.base_model.encoder

    res_encoder = encoder.forward(**data_tok,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=True)

    for a,b in zip(res_generate['encoder_hidden_states'], res_forward['encoder_hidden_states']):
        assert torch.all(torch.eq(a,b)).item()
        print(torch.all(torch.eq(a,b)).item())

    for a,b in zip(res_generate['encoder_hidden_states'], res_encoder['hidden_states']):
        assert torch.all(torch.eq(a,b)).item()
        print(torch.all(torch.eq(a,b)).item())