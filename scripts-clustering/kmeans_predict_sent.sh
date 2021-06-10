# fbase=wmt20.en-cs.en
# encoder_type=bert
# NUM_CLUSTERS=8
# experiments_basedir=/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments
# python scripts-clustering/kmeans_predict_sent.py \
#     ${experiments_basedir}/en_cs_ParaCrawl/outputs/bert/kmeans_${encoder_type}_sent_${NUM_CLUSTERS}.pkl \
#     ${experiments_basedir}/wmt-on-clusters/outputs/bert/sent_means_${fbase}.pkl \
#     ${experiments_basedir}/wmt-on-clusters/outputs/cluster_labels/${fbase}.${encoder_type}.sent.k_${NUM_CLUSTERS}.clustlabels


# fbase=wmt20.de-en.de
# encoder_type=bert
# NUM_CLUSTERS=8
# experiments_basedir=/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments
# kmdens_model_dir=de_en_ParaCrawl_3m
# python scripts-clustering/kmeans_predict_sent.py \
#     ${experiments_basedir}/${kmdens_model_dir}/outputs/bert/kmeans_${encoder_type}_sent_${NUM_CLUSTERS}.pkl \
#     ${experiments_basedir}/wmt-on-clusters/outputs/bert/sent_means_${fbase}.pkl \
#     ${experiments_basedir}/wmt-on-clusters/outputs/cluster_labels/${fbase}.${encoder_type}.sent.k_${NUM_CLUSTERS}.clustlabels




# fbase=wmt18.en-et.en
# encoder_type=bert
# NUM_CLUSTERS=8
# experiments_basedir=/gpfs/hpc/projects/nlpgroup/bergamot/da/experiments
# kmdens_model_dir=en_et_xlm-roberta-base/internals-docs
# python scripts-clustering/kmeans_predict_sent.py \
#     ${experiments_basedir}/${kmdens_model_dir}/kmeans_train_sent_8.pkl \
#     ${experiments_basedir}/wmt-on-clusters/outputs/bert/sent_means_${fbase}.pkl \
#     ${experiments_basedir}/wmt-on-clusters/outputs/cluster_labels/${fbase}.${encoder_type}.sent.k_${NUM_CLUSTERS}.clustlabels


