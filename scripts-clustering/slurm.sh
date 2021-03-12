#!/bin/bash

#SBATCH -t 192:00:00
#SBATCH --mem=90GB

#SBATCH -J en-et

module load python/3.6.3

source activate da2

EXP_NAME="bert"
#EXP_NAME="bert"
LANG_PAIR="en-et"
#SENT_OR_DOC="sent"
SENT_OR_DOC="doc"
NUM_CLUSTERS=8

FN="$EXP_NAME-$LANG_PAIR-$SENT_OR_DOC-c8.log"
echo "$FN"

python scripts-clustering/kmeans_train.py $EXP_NAME $LANG_PAIR $SENT_OR_DOC $NUM_CLUSTERS > "$FN"

python scripts-clustering/kmeans_predict.py $EXP_NAME $LANG_PAIR $SENT_OR_DOC $NUM_CLUSTERS >> "$FN"
