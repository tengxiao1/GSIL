export CUDA_VISIBLE_DEVICES=0,5,6,7
log_file=logs/zephyr_full_62k_greedy_generate.log

if [ ! -d "logs" ];then
    mkdir "logs"
fi

accelerate launch --main_process_port=2950 gsil/generate.py > ${log_file} 2>&1
