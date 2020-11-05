#!/bin/bash
​
#The name of the job is train
#SBATCH -J concat_base
​
#The job requires 1 compute node
#SBATCH -N 1
​
#The job requires 1 task per node
#SBATCH --ntasks-per-node=1
​
#The maximum walltime of the job is 8 days
#SBATCH -t 192:00:00
​
#SBATCH --mail-type=ALL
#SBATCH --mail-user=max.del.edu@gmail.com
​
#SBATCH --mem=20GB
​
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
​
module load python/3.6.3/CUDA
​
conda activate da
​
EXP_NAME="concat_base_big_batch_lr"
SAVE_DIR="experiments/"$EXP_NAME

fairseq-train \
    data-prep/bin-data-en-et-base \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 15000 \
    --log-format json \
    --save-dir $SAVE_DIR \
    --tensorboard-logdir $SAVE_DIR/"log-tb" \
    2>&1 | tee $SAVE_DIR/"log.out"
