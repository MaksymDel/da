# Translation Transformers Rediscover Inherent Data Domains

This repository contains clustering pipeline for [Translation Transformers Rediscover Inherent Data Domains](https://aclanthology.org/2021.wmt-1.65.pdf). 

See [this code repo](https://github.com/TartuNLP/inherent-domains-wmt21) for paper notebooks and NMT training scipts

Dir stucture:
```
* faieseq (clonned fairseq repo)
* data-prep (copy from `/gpfs/hpc/projects/nlpgroup/bergamot/data-prep`)
* experiments (for model checkpoints and tb log; subfolders include "concat", "finetuned_europarl", "domain_control", etc.)
* scripts (running commands for different DA scenarious and data prep)
```

## Setup
```bash
conda create -n da python=3.8
conda activate da
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install scipy numpy pandas transformers==4.0 sentencepiece tensorboardX

git clone https://github.com/maksym-del/da
cd da

git clone https://github.com/pytorch/fairseq
cd fairseq
# fairseq commit: 1a709b2a401ac8bd6d805c8a6a5f4d7f03b923ff
# git reset --hard 1a709b2a401ac8bd6d805c8a6a5f4d7f03b923ff
pip install --editable ./

pip install tensorboard
pip install tensorboardX

cd ..
cp -r /gpfs/hpc/projects/nlpgroup/bergamot/data-prep .
```

## Train 
```
bash scripts/SCRIPTNAME.sh
or 
sbatch scripts/SCRIPTNAME.slurm
```

## Using clustering scripts:
```
0) Before running below's scripts, change paths to data and models in them (see "CHANGE THIS LINE" comment)

1) Convert trained fseq Transfomer to Huggingface format:
python scripts-clustering/convert_chkp_fseq_to_hf.py de-en

2) Extract NMT sentence and document representations:
# sent
python scripts-clustering/extract_reps.py nmt sent test
python scripts-clustering/extract_reps.py nmt sent dev
python scripts-clustering/extract_reps.py nmt sent train

# doc
python scripts-clustering/extract_reps.py nmt doc test
python scripts-clustering/extract_reps.py nmt doc dev
python scripts-clustering/extract_reps.py nmt doc train

3) Get clusters:
# sent
python scripts-clustering/kmeans_train.py nmt sent 8

python scripts-clustering/kmeans_predict.py nmt sent 8 test
python scripts-clustering/kmeans_predict.py nmt sent 8 dev
python scripts-clustering/kmeans_predict.py nmt sent 8 train

# doc
python scripts-clustering/kmeans_train.py nmt doc 8

python scripts-clustering/kmeans_predict.py nmt doc 8 test
python scripts-clustering/kmeans_predict.py nmt doc 8 dev
python scripts-clustering/kmeans_predict.py nmt doc 8 train

```

4) Extract XLM-R sentence and document representations:
```
Same commands as before, just use "bert" instead of "nmt.
For example:
python scripts-clustering/extract_reps.py bert sent test
```

## Fine-tune
Now that you have cluster (domain) separated data, 
fine-tune an NMT baseline (from before).to each of the clusters and get results. 

```
Note: to use multidomain kmeans models you need scikit-learn==0.22.2.post1
```

Cite:
```
@inproceedings{del-etal-2021-translation,
    title = "Translation Transformers Rediscover Inherent Data Domains",
    author = "Del, Maksym  and
      Korotkova, Elizaveta  and
      Fishel, Mark",
    booktitle = "Proceedings of the Sixth Conference on Machine Translation",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.wmt-1.65",
    pages = "599--613",
    abstract = "Many works proposed methods to improve the performance of Neural Machine Translation (NMT) models in a domain/multi-domain adaptation scenario. However, an understanding of how NMT baselines represent text domain information internally is still lacking. Here we analyze the sentence representations learned by NMT Transformers and show that these explicitly include the information on text domains, even after only seeing the input sentences without domains labels. Furthermore, we show that this internal information is enough to cluster sentences by their underlying domains without supervision. We show that NMT models produce clusters better aligned to the actual domains compared to pre-trained language models (LMs). Notably, when computed on document-level, NMT cluster-to-domain correspondence nears 100{\%}. We use these findings together with an approach to NMT domain adaptation using automatically extracted domains. Whereas previous work relied on external LMs for text clustering, we propose re-using the NMT model as a source of unsupervised clusters. We perform an extensive experimental study comparing two approaches across two data scenarios, three language pairs, and both sentence-level and document-level clustering, showing equal or significantly superior performance compared to LMs.",
}
```
