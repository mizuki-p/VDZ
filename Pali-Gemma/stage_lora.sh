time=$(date "+%m_%d-%H:%M")

deepspeed --include localhost:6,7,8,9 \
    train_lora.py \
    --epoch         30 \
    --batchsize     4 \
    --gradient_acc  4 \
    --learning_rate 1e-5 \
    --dataset       vqa_rad \
    --split         train \
    --saving_path   outputs/vqarad_vdz \
    --pretrained_model_path outputs/vdz_base \
    --from_checkpoint False \
    --save_total_limit 1 \
    --saving_step   300 \
    --log_activate  True \
    --project_name  pali_gemma \
    --run_name      vqarad_vdz \
    --log_step      1 \
    --lr_scheduler_type 'cosine' \
    --warmup_ratio  0.03 \
    --seed          2024 \
    --data_seed     2024 \
    --bf16          True \
    --gradient_checkpointing True \
    --deepspeed deepspeed/config.json \
    --output_dir outputs/${time} \
    # --output_dir outputs/06_03-10:14
