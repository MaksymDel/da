MODEL_TYPE=bert
MODEL_NAME_OR_DIR=xlm-roberta-base

# filename=wmt20.en-cs.en

for filename in wmt18.en-et.en wmt20.en-cs.en wmt20.de-en.de
do
    FILE_TO_EMBED=experiments/wmt-on-clusters/${filename}
    SAVEFILE=experiments/wmt-on-clusters/outputs/$MODEL_TYPE/sent_means_${filename}.pkl

    python scripts-clustering/extract_reps_sent.py $MODEL_TYPE $MODEL_NAME_OR_DIR $FILE_TO_EMBED $SAVEFILE
done

#exp_folder = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/de_en_ParaCrawl_3m_20m_params"
#exp_folder = "/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/en_cs_ParaCrawl"

#data_exp_folder = exp_folder # if the data is in the same dir as hf models and savedir


# wmt18.en-et.en
# wmt20.en-cs.en
# wmt20.de-en.de

# sp-concat-wmt20.de-en.de
# sp-ParaCrawl-3m-wmt20.de-en.de
# sp-ParaCrawl-10m-wmt20.de-en.de
# sp-ParaCrawl-wmt18.en-et.en
# sp-ParaCrawl-wmt20.en-cs.en



# data_file_path = f"{data_exp_folder}/data/sp-cl-ParaCrawl.en-cs.docs.{split}.both"
# data_file_path = f"{data_exp_folder}/cl-ParaCrawl.en-cs.docs.{split}"


# data_folder = "/gpfs/hpc/nlpgroup/bergamot/da/experiments/wmt-on-clusters"
# data_file_name = "wmt18.en-et.en"
# data_file_path = f"{data_folder}/{data_file_name}" 

# os.makedirs(savedir, exist_ok=True)