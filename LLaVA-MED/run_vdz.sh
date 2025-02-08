time=$(date "+%m_%d-%H:%M")
deepspeed --include localhost:4,5 --master_port 29697 \
    train_vdz.py \
    --epoch         10 \
    --batchsize     4 \
    --gradient_acc  4 \
    --learning_rate 1e-5 \
    --dataset       ovqa \
    --saving_path   outputs/acc/ovqa_vdz_2l3 \
    --from_checkpoint False \
    --save_total_limit 4 \
    --saving_step   5000 \
    --log_activate  True \
    --project_name  llava-med \
    --run_name      ovqa_vdz_acc_2l3 \
    --log_step      1 \
    --lr_scheduler_type 'cosine' \
    --warmup_ratio  0.03 \
    --seed          2024 \
    --data_seed     2024 \
    --bf16          True \
    --gradient_checkpointing True \
    --deepspeed config/ds_config.json \
    --output_dir outputs/${time}
    # --output_dir outputs/temp
