#!/bin/bash

# 输入文件数组
input_files=(
    "/workspace/magpie/data/Qwen2-7B-Instruct_topp1_temp1_1719289179/Magpie_Qwen2-7B-Instruct_1000000_1719289179_ins_checkpoint.json"
    "/workspace/magpie/data/LLaMA3-iterative-DPO-final_topp1_temp1_1719202931/Magpie_LLaMA3-iterative-DPO-final_500000_1719202931_ins_checkpoint.json"
)

# 标记任务
tag_mission="all"  # 可以根据需要修改任务类型

# 设备ID
device="0"

# 模型路径
model_path="meta-llama/Meta-Llama-3-8B-Instruct"
guard_model_path="meta-llama/Meta-Llama-Guard-2-8B"
reward_model_path="sfairXC/FsfairX-LLaMA3-RM-v0.1"

# 其他参数
tensor_parallel=1
gpu_memory_utilization=0.95
batch_size=1000

# 循环处理每个输入文件
for input_file in "${input_files[@]}"; do
    if [ ! -f $input_file ]; then
        echo "[run_unitag.sh] Input file $input_file not found!"
        continue
    fi

    # 获取输入文件的路径
    job_path=$(dirname "$input_file")
    exec > >(tee -a "$job_path/tagging.log") 2>&1
    echo "[run_unitag.sh] Job Path: $job_path"
    echo "[run_unitag.sh] Input File: $input_file"
    echo "[run_unitag.sh] Tagging Mission: $tag_mission"
    echo "[run_unitag.sh] Model Name: $model_path"
    echo "[run_unitag.sh] System Config: device=$device, batch_size=$batch_size, tensor_parallel=$tensor_parallel"

    if [ $tag_mission == "difficulty" ] || [ $tag_mission == "all" ]; then
        echo "[run_unitag.sh] Start Generating Difficulty Tags..."
        CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
            --device $device \
            --model_path $model_path \
            --input_file $input_file \
            --tag_mission "difficulty" \
            --tensor_parallel $tensor_parallel \
            --gpu_memory_utilization $gpu_memory_utilization \
            --batch_size $batch_size \

        echo "[run_unitag.sh] Finish Generating Difficulty Tags!"

        # 更新 input_file 为生成的 difficulty 标签文件
        input_file_name=$(basename $input_file)
        input_file_dir=$(dirname $input_file)
        input_file_name_no_ext="${input_file_name%.*}"
        input_file_ext="${input_file_name##*.}"
        difficulty_tag_file="${input_file_dir}/${input_file_name_no_ext}_difficulty.${input_file_ext}"
        input_file=$difficulty_tag_file
        echo "[run_unitag.sh] Difficulty Tagged File: $input_file"
    fi

    if [ $tag_mission == "quality" ] || [ $tag_mission == "all" ]; then
        echo "[run_unitag.sh] Start Generating Quality Tags..."
        CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
            --device $device \
            --model_path $model_path \
            --input_file $input_file \
            --tag_mission "quality" \
            --tensor_parallel $tensor_parallel \
            --gpu_memory_utilization $gpu_memory_utilization \
            --batch_size $batch_size \

        echo "[run_unitag.sh] Finish Generating Quality Tags!"

        # 更新 input_file 为生成的 quality 标签文件
        input_file_name=$(basename $input_file)
        input_file_dir=$(dirname $input_file)
        input_file_name_no_ext="${input_file_name%.*}"
        input_file_ext="${input_file_name##*.}"
        quality_tag_file="${input_file_dir}/${input_file_name_no_ext}_quality.${input_file_ext}"
        input_file=$quality_tag_file
        echo "[run_unitag.sh] Quality Tagged File: $input_file"
    fi

    if [ $tag_mission == "classification" ] || [ $tag_mission == "all" ]; then
        echo "[run_unitag.sh] Start Generating Task Tags..."
        CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
            --device $device \
            --model_path $model_path \
            --input_file $input_file \
            --tag_mission "classification" \
            --tensor_parallel $tensor_parallel \
            --gpu_memory_utilization $gpu_memory_utilization \
            --batch_size $batch_size \

        echo "[run_unitag.sh] Finish Generating Task Tags!"

        # 更新 input_file 为生成的 classification 标签文件
        input_file_name=$(basename $input_file)
        input_file_dir=$(dirname $input_file)
        input_file_name_no_ext="${input_file_name%.*}"
        input_file_ext="${input_file_name##*.}"
        classification_tag_file="${input_file_dir}/${input_file_name_no_ext}_category.${input_file_ext}"
        input_file=$classification_tag_file
        echo "[run_unitag.sh] Task Tagged File: $input_file"
    fi

    if [ $tag_mission == "safety" ] || [ $tag_mission == "all" ]; then
        echo "[run_unitag.sh] Start Generating Safety Tags..."
        CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
            --device $device \
            --guard_model_path $guard_model_path \
            --input_file $input_file \
            --tag_mission "safety" \
            --tensor_parallel $tensor_parallel \
            --gpu_memory_utilization $gpu_memory_utilization \
            --batch_size $batch_size \

        echo "[run_unitag.sh] Finish Generating Safety Tags!"

        # 更新 input_file 为生成的 safety 标签文件
        input_file_name=$(basename $input_file)
        input_file_dir=$(dirname $input_file)
        input_file_name_no_ext="${input_file_name%.*}"
        input_file_ext="${input_file_name##*.}"
        safety_tag_file="${input_file_dir}/${input_file_name_no_ext}_safety.${input_file_ext}"
        input_file=$safety_tag_file
        echo "[run_unitag.sh] Safety Tagged File: $input_file"
    fi

    if [ $tag_mission == "reward" ] || [ $tag_mission == "all" ]; then
        echo "[run_unitag.sh] Start Generating Reward Tags..."
        python ../exp/unitag.py \
            --device $device \
            --reward_model_path $reward_model_path \
            --input_file $input_file \
            --tag_mission "reward" \
            --tensor_parallel $tensor_parallel \
            --batch_size 1 \

        echo "[run_unitag.sh] Finish Generating Reward Tags!"

        # 更新 input_file 为生成的 reward 标签文件
        input_file_name=$(basename $input_file)
        input_file_dir=$(dirname $input_file)
        input_file_name_no_ext="${input_file_name%.*}"
        input_file_ext="${input_file_name##*.}"
        reward_tag_file="${input_file_dir}/${input_file_name_no_ext}_reward.${input_file_ext}"
        input_file=$reward_tag_file
        echo "[run_unitag.sh] Reward Tagged File: $input_file"
    fi

    if [ $tag_mission == "language" ] || [ $tag_mission == "all" ]; then
        echo "[run_unitag.sh] Start Generating Language Tags..."
        CUDA_VISIBLE_DEVICES=$device python ../exp/unitag.py \
            --device $device \
            --input_file $input_file \
            --tag_mission "language" \

        echo "[run_unitag.sh] Finish Generating Language Tags!"

        # 更新 input_file 为生成的 language 标签文件
        input_file_name=$(basename $input_file)
        input_file_dir=$(dirname $input_file)
        input_file_name_no_ext="${input_file_name%.*}"
        input_file_ext="${input_file_name##*.}"
        language_tag_file="${input_file_dir}/${input_file_name_no_ext}_language.${input_file_ext}"
        input_file=$language_tag_file
        echo "[run_unitag.sh] Language Tagged File: $input_file"
    fi

    echo "[run_unitag.sh] Finish Tagging Mission: $tag_mission for file $input_file"
done

echo "[run_unitag.sh] All files processed."
