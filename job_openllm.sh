#!/bin/bash
#SBATCH --partition=M1
#SBATCH --qos=q_d8_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=30G
#SBATCH --job-name=GraphRAG-cs
#SBATCH --output=./logs/output_GraphRAG.out
#SBATCH --error=./logs/error_GraphRAG.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate digimon

# Step 1: 启动 ollama 服务（后台运行）
echo "Starting ollama service..."
ollama serve > ./logs/ollama_service.log 2>&1 &
SERVE_PID=$!
echo "ollama service started with PID: $SERVE_PID"

# 等待服务初始化（根据实际情况调整时间）
echo "Waiting for ollama service to initialize..."
sleep 10  # 初始等待

# Step 2: 运行 ollama run 命令（连接到已启动的服务）
echo "Starting ollama service..."
ollama run deepseek-r1:1.5b --port 11413 > ./logs/ollama.log 2>&1 &

# 保存ollama服务的PID，以便后续监控或终止
RUN_PID=$!
echo "ollama model loaded with PID: $RUN_PID"

# 等待ollama服务完全启动（根据你的模型大小调整等待时间）
echo "Waiting for ollama service to initialize..."
sleep 30  # 根据模型大小可能需要更长时间

# 启动主程序，向ollama服务发送消息并获取结果
echo "Starting main application..."
python main.py -opt Option/Method/RAPTOR.yaml -dataset_name cs > ./logs/app.log 2>&1

# 主程序完成后，终止ollama服务
# 清理资源
echo "Main application completed. Stopping ollama processes..."
kill -9 $RUN_PID $SERVE_PID

echo "Job completed!"