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

#SBATCH --exclude=falcon3

# srun -p gpu --gres gpu:tesla:1 -t 192:00:00 --exclude falcon5,falcon1,falcon2,falcon3 --pty bash

module load python/3.6.3/CUDA

source activate da

LANG_PAIR="en-et"
EXP_NAME="concat"
SAVE_DIR="experiments/$LANG_PAIR/$EXP_NAME"
DATA_PATH="data-prep/bin-data-en-et-base"

mkdir $SAVE_DIR

fairseq-train \
    $DATA_PATH \
    --arch transformer --max-epoch 60 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu --eval-bleu-remove-bpe=sentencepiece \
    --max-tokens 15000 \
    --log-format json \
    --save-dir $SAVE_DIR \
    --tensorboard-logdir $SAVE_DIR/"log-tb" \
    2>&1 | tee $SAVE_DIR/"log.out"