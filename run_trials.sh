export XLA_PYTHON_CLIENT_PREALLOCATE=false

# num_seeds=4
# for ((i = 0 ; i < $num_seeds ; i++ )); do 
#     CUDA_VISIBLE_DEVICES=1 python -m baselines.IPPO.ippo_ff_hanabi \
#         PROJECT="hanabi" \
#         WANDB_MODE="online" \
#         ENTITY=gyo \
#         WANDB_RUN_GROUP=hanabi_ippo_ff_token_obs_w_orig_obs_0 \
#         WANDB_RUN_NAME=ff_token_obs_w_orig_seed_$i \
#         SEED=$i \
#         USE_TOKEN_OBS=True
# done


for ((i = 0 ; i < 1 ; i++ )); do 
    CUDA_VISIBLE_DEVICES=1 python -m baselines.IPPO.ippo_transf_hanabi \
        PROJECT="hanabi" \
        WANDB_MODE="online" \
        ENTITY=gyo \
        WANDB_RUN_GROUP=seq_len_${SEQ_LEN}_temporal_embed \
        WANDB_RUN_NAME=transf_seq_len_${SEQ_LEN}_temporal_embed_seed_$i \
        SEQUENCE_LENGTH=${SEQ_LEN} \
        SEED=$i
done
        # WANDB_RUN_GROUP=test_seq_5_w_orig_obs \
        # WANDB_RUN_GROUP=hanabi_ippo_transf_token_obs_w_belief_v0 \