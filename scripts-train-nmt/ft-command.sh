#!/bin/bash

#The name of the job is train
#SBATCH -J docall

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is 8 days
#SBATCH -t 192:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=max.del.edu@gmail.com

#SBATCH --mem=20GB

#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

#SBATCH --exclude=falcon1,falcon2,falcon3


module load python/3.6.3/CUDA

source activate da


fairseq-preprocess --source-lang en --target-lang et \
    --trainpref train --validpref valid --testpref test \
    --destdir data-bin --joined-dictionary \
    --srcdict ../../../bin-data-en-et-base/dict.en.txt


fairseq-train \
    data-bin \
    --finetune-from-model ../../../en_et_concat60/checkpoint60.pt \
    --arch transformer --max-epoch 50 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.000125 --lr-scheduler reduce_lr_on_plateau --lr-patience 3 --lr-shrink 0.5 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu --eval-bleu-remove-bpe=sentencepiece \
    --max-tokens 15000 \
    --log-format json \
    --save-dir chkp \
    --tensorboard-logdir chkp/log-tb \
    2>&1 | tee chkp/log.out