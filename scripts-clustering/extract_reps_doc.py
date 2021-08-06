import os
import sys

sys.path.append('.')

from da.embed_utils import compute_doc_reps, extract_reps_sent, read_doc_indexed_data, pickle_dump_to_file


if __name__ == '__main__':
    encoder_type = sys.argv[1] # nmt / bert
    assert encoder_type in ["nmt", "bert"]


    data_doc_indiced_path = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/de_en_ParaCrawl_3m/cl-ParaCrawl.en-cs.docs.test"

    savedir = f"{data_exp_folder}/outputs/{encoder_type}"
    os.makedirs(savedir, exist_ok=True)
    filename_loadfile_sent_means = f"{savedir}/sent_means_{split}.pkl"
    filename_savefile_docmeans = f"{savedir}/doc_means_{split}.pkl"
    filename_savefile_docids = f"{savedir}/docids_{split}.pkl" # uniq docids

    _, doc_ids = read_doc_indexed_data(data_doc_indiced_path)
    del _

    encoded_sent = pickle_load_from_file(filename_loadfile_sent_means)

    # Compute doc embeddings
    encoded_doc, another_docids = compute_doc_reps(encoded_sent, doc_ids)

    pickle_dump_to_file(encoded_doc, filename_savefile_docmeans)
    pickle_dump_to_file(another_docids, filename_savefile_docids)
    