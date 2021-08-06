# DA

Note: to use multidomain kmeans models you need scikit-learn==0.22.2.post1

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
