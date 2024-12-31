time=$(date "+%m_%d-%H:%M")

deepspeed --include localhost:6,7,8,9 \
    train_lora.py \
    --epoch         50 \
    --batchsize     16 \
    --gradient_acc  1 \
    --learning_rate 1e-5 \
    --dataset       vqa_rad \
    --split         train \
    --saving_path   outputs/rad_vdz \
    --pretrained_model_path pretrained_models/vdz_base \
    --from_checkpoint False \
    --save_total_limit 1 \
    --saving_step   300 \
    --log_activate  True \
    --project_name  bridge_tower \
    --run_name      rad_vdz \
    --log_step      1 \
    --lr_scheduler_type 'cosine' \
    --warmup_ratio  0.03 \
    --seed          2024 \
    --data_seed     2024 \
    --bf16          True \
    --gradient_checkpointing False \
    --deepspeed deepspeed/config.json \
    --output_dir outputs/${time} \
    # --output_dir outputs/06_03-10:14
