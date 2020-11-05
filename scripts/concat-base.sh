SAVE_DIR="experiments/concat"

CUDA_VISIBLE_DEVICES=1 fairseq-train \
    data-prep/bin-data-en-et-base \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --save-dir $SAVE_DIR \
    --tensorboard-logdir $SAVE_DIR/"log-tb" \
