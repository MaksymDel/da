```
(da) maksym@mavs:~/research/da$ fairseq-train -h
usage: fairseq-train [-h] [--no-progress-bar] [--log-interval LOG_INTERVAL] [--log-format {json,none,simple,tqdm}] [--tensorboard-logdir TENSORBOARD_LOGDIR] [--wandb-project WANDB_PROJECT] [--seed SEED] [--cpu] [--tpu] [--bf16] [--memory-efficient-bf16] [--fp16] [--memory-efficient-fp16]
                     [--fp16-no-flatten-grads] [--fp16-init-scale FP16_INIT_SCALE] [--fp16-scale-window FP16_SCALE_WINDOW] [--fp16-scale-tolerance FP16_SCALE_TOLERANCE] [--min-loss-scale MIN_LOSS_SCALE] [--threshold-loss-scale THRESHOLD_LOSS_SCALE] [--user-dir USER_DIR]
                     [--empty-cache-freq EMPTY_CACHE_FREQ] [--all-gather-list-size ALL_GATHER_LIST_SIZE] [--model-parallel-size MODEL_PARALLEL_SIZE] [--quantization-config-path QUANTIZATION_CONFIG_PATH] [--profile] [--tokenizer {moses,nltk,space}]
                     [--bpe {hf_byte_bpe,fastbpe,sentencepiece,characters,subword_nmt,gpt2,bytes,byte_bpe,bert}]
                     [--criterion {composite_loss,cross_entropy,ctc,label_smoothed_cross_entropy,label_smoothed_cross_entropy_with_alignment,wav2vec,legacy_masked_lm_loss,nat_loss,masked_lm,sentence_prediction,sentence_ranking,adaptive_loss,vocab_parallel_cross_entropy}]
                     [--optimizer {adagrad,sgd,adam,nag,adadelta,adamax,lamb,adafactor}] [--lr-scheduler {inverse_sqrt,cosine,triangular,polynomial_decay,tri_stage,fixed,reduce_lr_on_plateau}] [--scoring {wer,sacrebleu,bleu,chrf}] [--task TASK] [--num-workers NUM_WORKERS]
                     [--skip-invalid-size-inputs-valid-test] [--max-tokens MAX_TOKENS] [--batch-size BATCH_SIZE] [--required-batch-size-multiple REQUIRED_BATCH_SIZE_MULTIPLE] [--required-seq-len-multiple REQUIRED_SEQ_LEN_MULTIPLE] [--dataset-impl {raw,lazy,cached,mmap,fasta}]
                     [--data-buffer-size DATA_BUFFER_SIZE] [--train-subset TRAIN_SUBSET] [--valid-subset VALID_SUBSET] [--validate-interval VALIDATE_INTERVAL] [--validate-interval-updates VALIDATE_INTERVAL_UPDATES] [--validate-after-updates VALIDATE_AFTER_UPDATES]
                     [--fixed-validation-seed FIXED_VALIDATION_SEED] [--disable-validation] [--max-tokens-valid MAX_TOKENS_VALID] [--batch-size-valid BATCH_SIZE_VALID] [--curriculum CURRICULUM] [--gen-subset GEN_SUBSET] [--num-shards NUM_SHARDS] [--shard-id SHARD_ID]
                     [--distributed-world-size DISTRIBUTED_WORLD_SIZE] [--distributed-rank DISTRIBUTED_RANK] [--distributed-backend DISTRIBUTED_BACKEND] [--distributed-init-method DISTRIBUTED_INIT_METHOD] [--distributed-port DISTRIBUTED_PORT] [--device-id DEVICE_ID] [--local-rank LOCAL_RANK]
                     [--distributed-no-spawn] [--ddp-backend {c10d,no_c10d}] [--bucket-cap-mb BUCKET_CAP_MB] [--fix-batches-to-gpus] [--find-unused-parameters] [--fast-stat-sync] [--broadcast-buffers] [--distributed-wrapper {DDP,SlowMo}] [--slowmo-momentum SLOWMO_MOMENTUM]
                     [--slowmo-algorithm SLOWMO_ALGORITHM] [--localsgd-frequency LOCALSGD_FREQUENCY] [--nprocs-per-node NPROCS_PER_NODE] [--pipeline-model-parallel] [--pipeline-balance PIPELINE_BALANCE] [--pipeline-devices PIPELINE_DEVICES] [--pipeline-chunks PIPELINE_CHUNKS]
                     [--pipeline-encoder-balance PIPELINE_ENCODER_BALANCE] [--pipeline-encoder-devices PIPELINE_ENCODER_DEVICES] [--pipeline-decoder-balance PIPELINE_DECODER_BALANCE] [--pipeline-decoder-devices PIPELINE_DECODER_DEVICES] [--pipeline-checkpoint {always,never,except_last}]
                     [--zero-sharding {none,os}] [--arch ARCH] [--max-epoch MAX_EPOCH] [--max-update MAX_UPDATE] [--stop-time-hours STOP_TIME_HOURS] [--clip-norm CLIP_NORM] [--sentence-avg] [--update-freq UPDATE_FREQ] [--lr LR] [--min-lr MIN_LR] [--use-bmuf] [--save-dir SAVE_DIR]
                     [--restore-file RESTORE_FILE] [--finetune-from-model FINETUNE_FROM_MODEL] [--reset-dataloader] [--reset-lr-scheduler] [--reset-meters] [--reset-optimizer] [--optimizer-overrides OPTIMIZER_OVERRIDES] [--save-interval SAVE_INTERVAL] [--save-interval-updates SAVE_INTERVAL_UPDATES]
                     [--keep-interval-updates KEEP_INTERVAL_UPDATES] [--keep-last-epochs KEEP_LAST_EPOCHS] [--keep-best-checkpoints KEEP_BEST_CHECKPOINTS] [--no-save] [--no-epoch-checkpoints] [--no-last-checkpoints] [--no-save-optimizer-state] [--best-checkpoint-metric BEST_CHECKPOINT_METRIC]
                     [--maximize-best-checkpoint-metric] [--patience PATIENCE] [--checkpoint-suffix CHECKPOINT_SUFFIX] [--checkpoint-shard-count CHECKPOINT_SHARD_COUNT]

optional arguments:
  -h, --help            show this help message and exit
  --no-progress-bar     disable progress bar
  --log-interval LOG_INTERVAL
                        log progress every N batches (when progress bar is disabled)
  --log-format {json,none,simple,tqdm}
                        log format to use
  --tensorboard-logdir TENSORBOARD_LOGDIR
                        path to save logs for tensorboard, should match --logdir of running tensorboard (default: no tensorboard logging)
  --wandb-project WANDB_PROJECT
                        Weights and Biases project name to use for logging
  --seed SEED           pseudo random number generator seed
  --cpu                 use CPU instead of CUDA
  --tpu                 use TPU instead of CUDA
  --bf16                use bfloat16; implies --tpu
  --memory-efficient-bf16
                        use a memory-efficient version of BF16 training; implies --bf16
  --fp16                use FP16
  --memory-efficient-fp16
                        use a memory-efficient version of FP16 training; implies --fp16
  --fp16-no-flatten-grads
                        don't flatten FP16 grads tensor
  --fp16-init-scale FP16_INIT_SCALE
                        default FP16 loss scale
  --fp16-scale-window FP16_SCALE_WINDOW
                        number of updates before increasing loss scale
  --fp16-scale-tolerance FP16_SCALE_TOLERANCE
                        pct of updates that can overflow before decreasing the loss scale
  --min-loss-scale MIN_LOSS_SCALE
                        minimum FP16 loss scale, after which training is stopped
  --threshold-loss-scale THRESHOLD_LOSS_SCALE
                        threshold FP16 loss scale from below
  --user-dir USER_DIR   path to a python module containing custom extensions (tasks and/or architectures)
  --empty-cache-freq EMPTY_CACHE_FREQ
                        how often to clear the PyTorch CUDA cache (0 to disable)
  --all-gather-list-size ALL_GATHER_LIST_SIZE
                        number of bytes reserved for gathering stats from workers
  --model-parallel-size MODEL_PARALLEL_SIZE
                        total number of GPUs to parallelize model over
  --quantization-config-path QUANTIZATION_CONFIG_PATH
                        path to quantization config file
  --profile             enable autograd profiler emit_nvtx
  --tokenizer {moses,nltk,space}
  --bpe {hf_byte_bpe,fastbpe,sentencepiece,characters,subword_nmt,gpt2,bytes,byte_bpe,bert}
  --criterion {composite_loss,cross_entropy,ctc,label_smoothed_cross_entropy,label_smoothed_cross_entropy_with_alignment,wav2vec,legacy_masked_lm_loss,nat_loss,masked_lm,sentence_prediction,sentence_ranking,adaptive_loss,vocab_parallel_cross_entropy}
  --optimizer {adagrad,sgd,adam,nag,adadelta,adamax,lamb,adafactor}
  --lr-scheduler {inverse_sqrt,cosine,triangular,polynomial_decay,tri_stage,fixed,reduce_lr_on_plateau}
  --scoring {wer,sacrebleu,bleu,chrf}
  --task TASK           task

dataset_data_loading:
  --num-workers NUM_WORKERS
                        how many subprocesses to use for data loading
  --skip-invalid-size-inputs-valid-test
                        ignore too long or too short lines in valid and test set
  --max-tokens MAX_TOKENS
                        maximum number of tokens in a batch
  --batch-size BATCH_SIZE, --max-sentences BATCH_SIZE
                        number of examples in a batch
  --required-batch-size-multiple REQUIRED_BATCH_SIZE_MULTIPLE
                        batch size will be a multiplier of this value
  --required-seq-len-multiple REQUIRED_SEQ_LEN_MULTIPLE
                        maximum sequence length in batch will be a multiplier of this value
  --dataset-impl {raw,lazy,cached,mmap,fasta}
                        output dataset implementation
  --data-buffer-size DATA_BUFFER_SIZE
                        Number of batches to preload
  --train-subset TRAIN_SUBSET
                        data subset to use for training (e.g. train, valid, test)
  --valid-subset VALID_SUBSET
                        comma separated list of data subsets to use for validation (e.g. train, valid, test)
  --validate-interval VALIDATE_INTERVAL
                        validate every N epochs
  --validate-interval-updates VALIDATE_INTERVAL_UPDATES
                        validate every N updates
  --validate-after-updates VALIDATE_AFTER_UPDATES
                        dont validate until reaching this many updates
  --fixed-validation-seed FIXED_VALIDATION_SEED
                        specified random seed for validation
  --disable-validation  disable validation
  --max-tokens-valid MAX_TOKENS_VALID
                        maximum number of tokens in a validation batch (defaults to --max-tokens)
  --batch-size-valid BATCH_SIZE_VALID, --max-sentences-valid BATCH_SIZE_VALID
                        batch size of the validation batch (defaults to --batch-size)
  --curriculum CURRICULUM
                        don't shuffle batches for first N epochs
  --gen-subset GEN_SUBSET
                        data subset to generate (train, valid, test)
  --num-shards NUM_SHARDS
                        shard generation over N shards
  --shard-id SHARD_ID   id of the shard to generate (id < num_shards)

distributed_training:
  --distributed-world-size DISTRIBUTED_WORLD_SIZE
                        total number of GPUs across all nodes (default: all visible GPUs)
  --distributed-rank DISTRIBUTED_RANK
                        rank of the current worker
  --distributed-backend DISTRIBUTED_BACKEND
                        distributed backend
  --distributed-init-method DISTRIBUTED_INIT_METHOD
                        typically tcp://hostname:port that will be used to establish initial connetion
  --distributed-port DISTRIBUTED_PORT
                        port number (not required if using --distributed-init-method)
  --device-id DEVICE_ID
                        which GPU to use (usually configured automatically)
  --local-rank LOCAL_RANK
                        which GPU to use (usually configured automatically)
  --distributed-no-spawn
                        do not spawn multiple processes even if multiple GPUs are visible
  --ddp-backend {c10d,no_c10d}
                        DistributedDataParallel backend
  --bucket-cap-mb BUCKET_CAP_MB
                        bucket size for reduction
  --fix-batches-to-gpus
                        don't shuffle batches between GPUs; this reduces overall randomness and may affect precision but avoids the cost of re-reading the data
  --find-unused-parameters
                        disable unused parameter detection (not applicable to no_c10d ddp-backend
  --fast-stat-sync      [deprecated] this is now defined per Criterion
  --broadcast-buffers   Copy non-trainable parameters between GPUs, such as batchnorm population statistics
  --distributed-wrapper {DDP,SlowMo}
                        DistributedDataParallel backend
  --slowmo-momentum SLOWMO_MOMENTUM
                        SlowMo momentum term; by default use 0.0 for 16 GPUs, 0.2 for 32 GPUs; 0.5 for 64 GPUs, 0.6 for > 64 GPUs
  --slowmo-algorithm SLOWMO_ALGORITHM
                        whether to use LocalSGD or SGP
  --localsgd-frequency LOCALSGD_FREQUENCY
                        Local SGD allreduce frequency
  --nprocs-per-node NPROCS_PER_NODE
                        number of GPUs in each node. An allreduce operation across GPUs in a node is very fast. Hence, we do allreduce across GPUs in a node, and gossip across different nodes
  --pipeline-model-parallel
                        if set, use pipeline model parallelism across GPUs
  --pipeline-balance PIPELINE_BALANCE
                        partition the model into N_K pieces, where each piece contains N_i layers. The sum(args.pipeline_balance) should equal the total number of layers in the model
  --pipeline-devices PIPELINE_DEVICES
                        a list of device indices indicating which device to place each of the N_K partitions. The length of this list should equal the length of the --pipeline-balance argument
  --pipeline-chunks PIPELINE_CHUNKS
                        microbatch count for pipeline model parallelism
  --pipeline-encoder-balance PIPELINE_ENCODER_BALANCE
                        partition the pipeline parallel encoder into N_K pieces, where each piece contains N_i layers. The sum(args.pipeline_encoder_balance) should equal the total number of encoder layers in the model
  --pipeline-encoder-devices PIPELINE_ENCODER_DEVICES
                        a list of device indices indicating which device to place each of the N_K partitions. The length of this list should equal the length of the --pipeline-encoder-balance argument
  --pipeline-decoder-balance PIPELINE_DECODER_BALANCE
                        partition the pipeline parallel decoder into N_K pieces, where each piece contains N_i layers. The sum(args.pipeline_decoder_balance) should equal the total number of decoder layers in the model
  --pipeline-decoder-devices PIPELINE_DECODER_DEVICES
                        a list of device indices indicating which device to place each of the N_K partitions. The length of this list should equal the length of the --pipeline-decoder-balance argument
  --pipeline-checkpoint {always,never,except_last}
                        checkpointing mode for pipeline model parallelism
  --zero-sharding {none,os}
                        ZeRO sharding

Model configuration:
  --arch ARCH, -a ARCH  model architecture

optimization:
  --max-epoch MAX_EPOCH
                        force stop training at specified epoch
  --max-update MAX_UPDATE
                        force stop training at specified update
  --stop-time-hours STOP_TIME_HOURS
                        force stop training after specified cumulative time (if >0)
  --clip-norm CLIP_NORM
                        clip threshold of gradients
  --sentence-avg        normalize gradients by the number of sentences in a batch (default is to normalize by number of tokens)
  --update-freq UPDATE_FREQ
                        update parameters every N_i batches, when in epoch i
  --lr LR               learning rate for the first N epochs; all epochs >N using LR_N (note: this may be interpreted differently depending on --lr-scheduler)
  --min-lr MIN_LR       stop training when the learning rate reaches this minimum
  --use-bmuf            specify global optimizer for syncing models on different GPUs/shards

checkpoint:
  --save-dir SAVE_DIR   path to save checkpoints
  --restore-file RESTORE_FILE
                        filename from which to load checkpoint (default: <save-dir>/checkpoint_last.pt
  --finetune-from-model FINETUNE_FROM_MODEL
                        finetune from a pretrained model; note that meters and lr scheduler will be reset
  --reset-dataloader    if set, does not reload dataloader state from the checkpoint
  --reset-lr-scheduler  if set, does not load lr scheduler state from the checkpoint
  --reset-meters        if set, does not load meters from the checkpoint
  --reset-optimizer     if set, does not load optimizer state from the checkpoint
  --optimizer-overrides OPTIMIZER_OVERRIDES
                        a dictionary used to override optimizer args when loading a checkpoint
  --save-interval SAVE_INTERVAL
                        save a checkpoint every N epochs
  --save-interval-updates SAVE_INTERVAL_UPDATES
                        save a checkpoint (and validate) every N updates
  --keep-interval-updates KEEP_INTERVAL_UPDATES
                        keep the last N checkpoints saved with --save-interval-updates
  --keep-last-epochs KEEP_LAST_EPOCHS
                        keep last N epoch checkpoints
  --keep-best-checkpoints KEEP_BEST_CHECKPOINTS
                        keep best N checkpoints based on scores
  --no-save             don't save models or checkpoints
  --no-epoch-checkpoints
                        only store last and best checkpoints
  --no-last-checkpoints
                        don't store last checkpoints
  --no-save-optimizer-state
                        don't save optimizer-state as part of checkpoint
  --best-checkpoint-metric BEST_CHECKPOINT_METRIC
                        metric to use for saving "best" checkpoints
  --maximize-best-checkpoint-metric
                        select the largest metric value for saving "best" checkpoints
  --patience PATIENCE   early stop training if valid performance doesn't improve for N consecutive validation runs; note that this is influenced by --validate-interval
  --checkpoint-suffix CHECKPOINT_SUFFIX
                        suffix to add to the checkpoint file name
  --checkpoint-shard-count CHECKPOINT_SHARD_COUNT
                        Number of shards containing the checkpoint - if the checkpoint is over 300GB, it is preferable to split it into shards to prevent OOM on CPU while loading the checkpoint
```



Also:
```
# @package _group_
common:
    no_progress_bar: false
    log_interval: 100
    log_format: null
    tensorboard_logdir: null
    wandb_project: null
    seed: 1
    cpu: false
    tpu: false
    bf16: false
    fp16: false
    memory_efficient_fp16: false
    memory_efficient_bf16: false
    fp16_no_flatten_grads: false
    fp16_init_scale: 128
    fp16_scale_window: null
    fp16_scale_tolerance: 0.0
    min_loss_scale: 1.0e-4
    threshold_loss_scale: null
    user_dir: null
    empty_cache_freq: 0
    all_gather_list_size: 16384
    model_parallel_size: 1
    quantization_config_path: null
    profile: false
distributed_training:
    distributed_rank: 0
    distributed_backend: "nccl"
    distributed_init_method: null
    distributed_port: -1
    device_id: 0
    local_rank: 0
    distributed_no_spawn: false
    ddp_backend: "c10d"
    bucket_cap_mb: 25
    fix_batches_to_gpus: false
    find_unused_parameters: false
    fast_stat_sync: false
    broadcast_buffers: false
    distributed_wrapper: "DDP"
    slowmo_momentum: null
    slowmo_algorithm: "LocalSGD"
    localsgd_frequency: 3
dataset:
    num_workers: 1
    skip_invalid_size_inputs_valid_test: false
    max_tokens: null
    batch_size: null
    required_batch_size_multiple: 8
    dataset_impl: null
    data_buffer_size: 10
    train_subset: "train"
    valid_subset: "valid"
    validate_interval: 1
    fixed_validation_seed: null
    disable_validation: false
    curriculum: 0
    gen_subset: "test"
    num_shards: 1
    shard_id: 0
    max_tokens_valid: ${dataset.max_tokens}
    batch_size_valid: ${dataset.batch_size}
optimization:
    max_epoch: 0
    max_update: 0
    clip_norm: 0.0
    sentence_avg: false
    update_freq: [ 1 ]
    lr: [ 0.25 ]
    min_lr: -1.0
    use_bmuf: false
checkpoint:
    save_dir: "checkpoints"
    restore_file: "checkpoint_last.pt"
    reset_dataloader: false
    reset_lr_scheduler: false
    reset_meters: false
    reset_optimizer: false
    optimizer_overrides: "{}"
    save_interval: 1
    save_interval_updates: 0
    keep_interval_updates: -1
    keep_last_epochs: -1
    keep_best_checkpoints: -1
    no_save: false
    no_epoch_checkpoints: false
    no_last_checkpoints: false
    no_save_optimizer_state: false
    best_checkpoint_metric: "loss"
    maximize_best_checkpoint_metric: false
    patience: -1
    checkpoint_suffix: ""
bmuf:
    block_lr: 1
    block_momentum: 0.875
    global_sync_iter: 50
    warmup_iterations: 500
    use_nbm: false
    average_sync: false
defaults:
    - task: language_modeling
    - model: null
    - criterion: null
    - optimizer: null
    - lr_scheduler: null
    - bpe: null
    - tokenizer: null
    - scoring: null
    - generation: null
    - common_eval: null
    - eval_lm: null
```