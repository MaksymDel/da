import os
import sys

sys.path.append('.')

from transformers import AutoModel, AutoTokenizer
from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.embed_utils import extract_reps_sent, read_doc_indexed_data, pickle_dump_to_file


# python scripts-clustering/extract_reps_sent.py xlm-roberta-base experiments/wmt-on-clusters/wmt18.en-et.en
if __name__ == '__main__':
    encoder_type = sys.argv[1] # nmt / bert
    model_name_or_dir = sys.argv[2]
    data_file_path = sys.argv[3]
    savefile = sys.argv[4]

    assert encoder_type in ["nmt", "bert"]

    # read data
    data, _ = read_doc_indexed_data(data_file_path)

    # setup model
    if encoder_type == 'nmt':
        #model_name_or_dir = f'{exp_folder}/hf'
        
        BATCH_SIZE = 2000
        LAYER_ID = 4

        tokenizer_hf = FSMTTokenizer.from_pretrained(model_name_or_dir)
        model_hf = FSMTForConditionalGeneration.from_pretrained(model_name_or_dir)
        model_hf = model_hf.cuda()
        encoder_hf = model_hf.base_model.encoder
        encoder_hf.device = model_hf.device

    elif encoder_type == 'bert':
        #model_name_or_dir = 'xlm-roberta-base'

        BATCH_SIZE = 2000 # probably can do 512
        LAYER_ID = 7

        tokenizer_hf = AutoTokenizer.from_pretrained(model_name_or_dir)
        encoder_hf = AutoModel.from_pretrained(model_name_or_dir)
        encoder_hf = encoder_hf.cuda()


    encoded_sent = extract_reps_sent(data=data,
                                    tokenizer_hf=tokenizer_hf, 
                                    encoder_hf=encoder_hf, 
                                    batch_size=BATCH_SIZE, 
                                    layer_id=LAYER_ID)

    pickle_dump_to_file(encoded_sent, savefile)


