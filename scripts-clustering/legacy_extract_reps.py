import os
import sys

sys.path.append('.')

from transformers import AutoModel, AutoTokenizer
from da.fsmt.modeling_fsmt import FSMTForConditionalGeneration
from da.fsmt.tokenization_fsmt import FSMTTokenizer
from da.embed_utils import extract_reps_doc, extract_reps_sent


def setup_model(encoder_type, exp_folder, split, data_exp_folder=None):
    # setup inputs
    if encoder_type == 'nmt':
        # TODO: move model creation and hparams back to the extract_sent_reps method? 
        # it works if i embed one file at a time; also have to pass cuda param then
        # not worth it at this point
        BATCH_SIZE = 2000
        LAYER_ID = 4
        model_name_or_dir = f'{exp_folder}/hf'

        tokenizer_hf = FSMTTokenizer.from_pretrained(model_name_or_dir)
        model_hf = FSMTForConditionalGeneration.from_pretrained(model_name_or_dir)
        model_hf = model_hf.cuda()
        encoder_hf = model_hf.base_model.encoder
        encoder_hf.device = model_hf.device

        # sp data
        if data_exp_folder is None:
            filename_data_doc_indexed = f"{exp_folder}/data/sp-cl-ParaCrawl.en-cs.docs.{split}.both"
        else:
            filename_data_doc_indexed = f"{data_exp_folder}/data/sp-cl-ParaCrawl.en-cs.docs.{split}.both"


    elif encoder_type == 'bert':
        BATCH_SIZE = 2000 # probably can do 512
        LAYER_ID = 7
        model_name_or_dir = 'xlm-roberta-base'

        tokenizer_hf = AutoTokenizer.from_pretrained(model_name_or_dir)
        encoder_hf = AutoModel.from_pretrained(model_name_or_dir)
        encoder_hf = encoder_hf.cuda()

        # no sp data

        if data_exp_folder is None:
            filename_data_doc_indexed = f"{exp_folder}/cl-ParaCrawl.en-cs.docs.{split}"
        else:
            filename_data_doc_indexed = f"{data_exp_folder}/cl-ParaCrawl.en-cs.docs.{split}"

    return {
        'tokenizer_hf': tokenizer_hf, 
        'encoder_hf': encoder_hf, 
        'layer_id': LAYER_ID, 
        'batch_size': BATCH_SIZE, 
        'filename_data_doc_indexed': filename_data_doc_indexed
        }



if __name__ == '__main__':
    encoder_type = sys.argv[1] # nmt / bert
    sent_or_doc = sys.argv[2] # sent / doc
    split = sys.argv[3] # train / dev / test

    assert encoder_type in ["nmt", "bert"]
    assert sent_or_doc in ["sent", "doc"]
    assert split in ["train", "dev", "test", "dev-cl", "test-cl"]

    # CHANGE THIS LINES
    #exp_folder = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/de_en_ParaCrawl_3m_20m_params"
    exp_folder = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/en_cs_ParaCrawl"
    #data_exp_folder = exp_folder # if the data is in the same dir as hf models and savedir
    #data_exp_folder = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/de_en_ParaCrawl_3m"
    data_exp_folder = "/gpfs/hpc/projects/nlpgroup/bergamot/paracrawl-cz-en/"
    
    # for doc-reps, only for docids
    filename_data_doc_indexed = f"{data_exp_folder}/cl-ParaCrawl.en-cs.docs.{split}" # only for docids

    #

    savedir = f"{exp_folder}/outputs/{encoder_type}"
    os.makedirs(savedir, exist_ok=True)
    
    # setup inputs    
    if sent_or_doc == "sent":
        args = setup_model(encoder_type, exp_folder, split, data_exp_folder)

        args["filename_savefile"] = f"{savedir}/sent_means_{split}.pkl" 

        extract_reps_sent(**args)

    elif sent_or_doc == 'doc':
        filename_data_doc_indexed = filename_data_doc_indexed # for clarity
        filename_loadfile_sent_means = f"{savedir}/sent_means_{split}.pkl"
        filename_savefile_docmeans = f"{savedir}/doc_means_{split}.pkl"
        filename_savefile_docids = f"{savedir}/docids_{split}.pkl" # uniq docids

        extract_reps_doc(
            filename_data_doc_indexed=filename_data_doc_indexed,
            filename_sent_means=filename_loadfile_sent_means,
            filename_savefile_doc_means=filename_savefile_docmeans,
            filename_savefile_doc_ids=filename_savefile_docids
            )
    
    else:
        raise ValueError("Wrong argument")

    #extract_reps_doc_sent(**args)
    #extract_reps_doc_given_sent(**args)