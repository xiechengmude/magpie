#/workspace/magpie/data/Qwen2-72B-Instruct_topp1_temp1_1719241315/Magpie_Qwen2-72B-Instruct_500000_1719241315_ins.json
model_path=${1:-"Qwen/Qwen2-72B-Instruct"}
total_prompts=${2:-100000}
ins_topp=${3:-1}
ins_temp=${4:-1}
res_topp=${5:-1}
res_temp=${6:-0}
res_rep=1
device="0,1,2,3"
tensor_parallel=4
gpu_memory_utilization=0.95
n=200
batch_size=1
API_KEY="dcb755c19ca4a0b591c8911ddd3ec5b5b3622e16d52e9e933edd1fedae78d423"
API_URL="https://api.together.xyz/v1/chat/completions"

echo "[magpie.sh] Start Generating Responses..."
CUDA_VISIBLE_DEVICES='' python ../exp/gen_res.py \
    --device $device \
    --model_path $model_path \
    --batch_size $batch_size \
    --top_p $res_topp \
    --temp $res_temp \
    --rep $res_rep \
    --api True \
    --api_url $API_URL \
    --api_key $API_KEY \
    --input_file /workspace/magpie/data/Qwen2-72B-Instruct_topp1_temp1_1719241315/Magpie_Qwen2-72B-Instruct_500000_1719241315_ins.json 

echo "[magpie.sh] Finish Generating Responses!"