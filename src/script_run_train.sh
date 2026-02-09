
TASK_NAME="task_name"

OUTPUT_DIR="exps/${TASK_NAME}"


OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=1 \
    --master_port 12322 --nnodes=1  --node_rank=0 --master_addr="127.0.0.1" \
    run_model.py \
    --task_name ${TASK_NAME} \
    --data_dir '/to/your/data/path' \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --val_batch_size 2 \
    --embed_dim 128 \
    --save_ckpt_freq 1 \
    --val_years 2023 \
    --test_years 2022 \
    --tp_var gpm_tp_24hr \
    --idx_of_day 1 3 \
    --opt adamw \
    --lr 1e-5 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --warmup_epochs 3 \
    --epochs 100 \

# idx_of_day: [0: 6h, 1: 12h, 2: 18h, 3: 24h(1d)]