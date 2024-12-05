model_path=alignment-handbook/zephyr-7b-sft-full
data_path=./data/iter1/train_data # training data dir path by combine.py
output_dir=./outputs_models/zephyr_full_62k_greedy_bernoulli_ao01s2 # model save path
log_file=logs/zephyr_full_62k_greedy_bernoulli_ao01s2.log # log save path

if [ ! -d "logs" ];then
    mkdir "logs"
fi

if [ ! -d "outputs_models" ];then
    mkdir "outputs_models"
fi

if [ ! -d ${output_dir} ];then
    mkdir ${output_dir}
fi

deepspeed --include localhost:0,5,6,7 --master_port 61000  gsil/run_gsil.py \
    --model_name_or_path ${model_path}\
    --torch_dtype "bfloat16" \
    --use_flash_attention_2 True \
    --dataset_path ${data_path} \
    --dataset_weight 1.0 \
    --dataset_splits "train" \
    --preprocessing_num_workers 4 \
    --bf16 True \
    --ddp_timeout 5400 \
    --loss_type "bernoulli_scale_shift"\
    --alpha 0.01 \
    --beta 2 \
    --do_eval False \
    --evaluation_strategy "no" \
    --hub_model_id "zephyr-7b-sft-full" \
    --learning_rate 5.0e-7 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --log_level "info" \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --lr_scheduler_type "linear"\
    --max_length 1024 \
    --max_prompt_length 512 \
    --num_train_epochs 1 \
    --optim rmsprop \
    --output_dir ${output_dir} \
    --deepspeed configs/deepspeed_config_bf16.json \
    --push_to_hub False \
    --save_strategy "epoch" \
    --save_total_limit 6 \
    --seed 42 \
    --warmup_steps 30 \
    --warmup_ratio 0.1 > ${log_file}  2>&1 
