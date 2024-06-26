model_path=${1:-"xDAN2099/xDAN-L3-MoE-Performance-RLHF-0416"}
total_prompts=${2:-1000000}
ins_topp=${3:-1}
ins_temp=${4:-1}
res_topp=${5:-1}
res_temp=${6:-0}
res_rep=1
device="0,1,2,3"
tensor_parallel=4
gpu_memory_utilization=0.95
n=200
batch_size=20

# Get Current Time
timestamp=$(date +%s)

# Generate Pretty Name
job_name="${model_path##*/}_topp${ins_topp}_temp${ins_temp}_${timestamp}"

### Setup Logging
log_dir="data"
if [ ! -d "../${log_dir}" ]; then
    mkdir -p "../${log_dir}"
fi

job_path="../${log_dir}/${job_name}"

mkdir -p $job_path
exec > >(tee -a "$job_path/${job_name}.log") 2>&1
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] Pretty name: $job_name"
echo "[magpie.sh] Total Prompts: $total_prompts"
echo "[magpie.sh] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp"
echo "[magpie.sh] Response Generation Config: temp=$res_temp, top_p=$res_topp, rep=$res_rep"
echo "[magpie.sh] System Config: device=$device, n=$n, batch_size=$batch_size, tensor_parallel=$tensor_parallel"
echo "[magpie.sh] Timestamp: $timestamp"
echo "[magpie.sh] Job Name: $job_name"

echo "[magpie.sh] Start Generating Instructions..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_ins.py \
    --device $device \
    --model_path $model_path \
    --total_prompts $total_prompts \
    --top_p $ins_topp \
    --temp $ins_temp \
    --tensor_parallel $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --n $n \
    --job_name $job_name \
    --timestamp $timestamp \

echo "[magpie.sh] Finish Generating Instructions!"

echo "[magpie.sh] Start Generating Responses..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_res.py \
    --model_path microsoft/WizardLM-2-8x22B \
    --batch_size 10 \
    --top_p 1 \
    --temp 0 \
    --rep 1 \
    --api True \
    --api_url https://api.together.xyz/v1/chat/completions \
    --api_key dcb755c19ca4a0b591c8911ddd3ec5b5b3622e16d52e9e933edd1fedae78d423 \
    --input_file /workspace/magpie/data/xDAN-L3-MoE-Performance-RLHF-0416_topp1_temp1_1719421016/Magpie_xDAN-L3-MoE-Performance-RLHF-0416_1000_1719421016_ins.json


echo "[magpie.sh] Finish Generating Responses!"