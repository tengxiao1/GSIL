#!/bin/bash
# 获取当前主机的索引


# 模型路径作为第一个参数传入脚本
model_path=$1
host_index=$2
model_name=$3
#nums_gpu=$4
echo $host_index > host_index.txt
#echo $nums_gpu
output_base_path=../outputs
# 根据任务执行相应的命令
torch_dtype=bfloat16
case $host_index in
    0)
        log_file="logs/eval_arc_$model_name.log"
        echo $log_file > log_file.txt
        nohup python3 main.py \
            --model hf-causal-experimental \
            --model_args pretrained=${model_path},dtype=${torch_dtype},use_accelerate=True\
            --num_fewshot 25 \
            --tasks arc_challenge \
            --no_cache \
            --output_base_path ${output_base_path} \
            --batch_size 10 > $log_file 2>&1 &
        ;;
    1)
        log_file="logs/eval_gsm8k_$model_name.log"
        echo $log_file > log_file.txt
        nohup python3 main.py \
            --model hf-causal-experimental \
            --model_args pretrained=${model_path},dtype=${torch_dtype},use_accelerate=True \
            --num_fewshot 5 \
            --tasks gsm8k \
            --no_cache \
            --output_base_path ${output_base_path} \
            --batch_size 8 > $log_file 2>&1 &
        ;;

    2)
        log_file="logs/eval_winogrande_$model_name.log"
        echo $log_file > log_file.txt
        nohup python3 main.py \
            --model hf-causal-experimental \
            --model_args pretrained=${model_path},dtype=${torch_dtype},use_accelerate=True \
            --num_fewshot 5 \
            --tasks winogrande \
            --output_base_path ${output_base_path} \
            --no_cache \
            --batch_size 10 > $log_file 2>&1 &
        ;;
    3)
        log_file="logs/eval_truthfulqa_$model_name.log"
        echo $log_file > log_file.txt
        nohup python3 main.py \
            --model hf-causal-experimental \
            --model_args pretrained=${model_path},dtype=${torch_dtype},use_accelerate=True \
            --tasks truthfulqa_mc \
            --no_cache \
            --output_base_path ${output_base_path} \
            --batch_size 1 > $log_file 2>&1 &
        ;;
    4)
        log_file="logs/eval_hellaswag_$model_name.log"
        echo $log_file > log_file.txt
        nohup python3 main.py \
            --model hf-causal-experimental \
            --model_args pretrained=${model_path},dtype=${torch_dtype},use_accelerate=True \
            --num_fewshot 10 \
            --tasks hellaswag \
            --no_cache \
            --output_base_path ${output_base_path} \
            --batch_size 50 > $log_file 2>&1 &
        ;;
    5)
        log_file="logs/eval_mmlu_$model_name.log"
        echo $log_file > log_file.txt
        nohup python3 main.py \
            --model hf-causal-experimental \
            --model_args pretrained=${model_path},dtype=${torch_dtype},use_accelerate=True \
            --num_fewshot 5 \
            --no_cache \
            --tasks  hendrycksTest-* \
            --output_base_path ${output_base_path} \
            --batch_size 10 > $log_file 2>&1 &
        ;;
    *)
        echo "No task assigned for host $host_index"
        ;;
esac
