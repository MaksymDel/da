DATA_PATH=experiments/EMEA-fix/concat-docs-data
RESULTS_PATH=experiments/res
set=test-cl

srclang=en
tgtlang=et


CLUST_NUM=3
EXP_NAME=CLUSTER_${CLUST_NUM}
SAVE_DIR=experiments/${srclang}_${tgtlang}_${EXP_NAME}
corpus=EMEA

model_path=/gpfs/hpc/projects/nlpgroup/bergamot/models/en_et_bert_clusters_doc_4_cluster${CLUST_NUM}_ft/checkpoint_best_dev_bleu.pt
bin_path=/gpfs/hpc/projects/nlpgroup/bergamot/en-et-concat-ft-bin-data/bin-data-en-et-bert_clusters_doc_4-cluster${CLUST_NUM}-ft/

# translate

cat $DATA_PATH/sp-cl-$corpus.$srclang-$tgtlang.docs.$set.bert_clusters_doc_emea_4_cluster${CLUST_NUM}.$srclang \
| fairseq-interactive $bin_path \
    --source-lang $srclang --target-lang $tgtlang \
    --path $model_path \
    --buffer-size 2000 --batch-size 32 --beam 5 \
> $RESULTS_PATH/transl_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.sys

# grep translations from the output file
grep "^H" $RESULTS_PATH/transl_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.sys | cut -f3 > $RESULTS_PATH/hyp_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.txt
# grep "^T" $RESULTS_PATH/out_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.sys | cut -f2 > $RESULTS_PATH/tgt_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.txt

# de-sentencepiece
python3 scripts-clustering/apply_sentencepiece.py --corpora experiments/res/sp.test-cl.emea.et --model /gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/EMEA-fix/concat-docs-data/preproc-models/fs-en-et --action restore

python3 scripts-clustering/apply_sentencepiece.py --corpora experiments/res/sp-cl-EMEA.en-et.docs.test-cl.bert_clusters_doc_emea_4_clusterorder.et --model /gpfs/hpc/projects/nlpgroup/bergamot/da/experiments/EMEA-fix/concat-docs-data/preproc-models/fs-en-et --action restore

# calculate bleu w/sacrebleu
echo $EXP_NAME
echo $corpus
cat $RESULTS_PATH/de-fs-en-et-hyp_${srclang}_${tgtlang}_${EXP_NAME}_${corpus}.txt | sacrebleu single-domain/sys-clusters/cl-$corpus.$srclang-$tgtlang.docs.$set.$tgtlang

sp-cl-EMEA.en-et.docs.test-cl.bert_clusters_doc_emea_4_clusterorder.et