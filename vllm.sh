#!/bin/bash
#
# --- SBATCH 配置 (与之前相同) ---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/vllm.out
#SBATCH -e logs/vllm.err
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --mem=512G

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# 2. 获取当前计算节点的 IP 地址
# hostname -I 通常返回节点的 IP，awk '{print $1}' 取第一个 IP
export HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
export PORT=8090

echo "------------------------------------------------"
echo "vLLM 正在节点 $HEAD_NODE_IP 上启动..."
echo "------------------------------------------------"

# 3. 将 API 地址写入共享文件，供 Streamlit 读取
# 格式如: http://10.10.10.5:8090/v1
echo "http://${HEAD_NODE_IP}:${PORT}/v1" > /home/yfjin/ClientStimul/run/api_info.txt

# 4. 启动 vLLM
# 注意 --host 0.0.0.0 允许外部连接
python -m vllm.entrypoints.openai.api_server \
    --model /home/yfjin/ClientStimul/Qwen2.5-7B-ClientStimul-Merged \
    --served-model-name client-stimul \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4720 \
    --trust-remote-code