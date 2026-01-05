# 通用基础阶段
FROM python:3.10-slim AS base

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y wget curl git sudo && \
    rm -rf /var/lib/apt/lists/*

# 创建目录
RUN mkdir -p /app/models /app/scripts

# 安装Python依赖
COPY ./src/flask_server_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 下载/复制 RKLLM 运行时库
COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so

# 复制频率修复脚本（仅作备份，不在构建时运行）
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh

# --- 模型镜像构建 ---

# 1. DeepSeek-R1-Distill-Qwen-1.5B_FP16
FROM base AS deepseek-r1-distill-qwen-1.5b-fp16-rk3576
# 直接复制预转换好的模型文件
RUN wget https://huggingface.co/JiahaoLi/DeepSeek-R1-Distill-Qwen-RK3576/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_G128_RK3576.rkllm -o /app/models/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_G128_RK3576.rkllm
COPY ./src/flask_server.py /app/
EXPOSE 8080
CMD ["python", "/app/flask_server.py", "--rkllm_model_path", "/app/models/DeepSeek-R1-Distill-Qwen-1.5B_FP16_RK3576.rkllm", "--target_platform", "rk3576"]

# 2. DeepSeek-R1-Distill-Qwen-1.5B_W4A16
FROM base AS deepseek-r1-distill-qwen-1.5b-w4a16-rk3576
RUN wget https://huggingface.co/JiahaoLi/DeepSeek-R1-Distill-Qwen-RK3576/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3576.rkllm -o /app/models/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3576.rkllm
COPY ./src/flask_server.py /app/
EXPOSE 8080
CMD ["python", "/app/flask_server.py", "--rkllm_model_path", "/app/models/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3576.rkllm", "--target_platform", "rk3576"]

# 3. DeepSeek-R1-Distill-Qwen-7B_W4A16
FROM base AS deepseek-r1-distill-qwen-7b-w4a16-rk3576
RUN wget https://huggingface.co/JiahaoLi/DeepSeek-R1-Distill-Qwen-RK3576/resolve/main/DeepSeek-R1-Distill-Qwen-7B_W4A16_RK3576.rkllm -o /app/models/DeepSeek-R1-Distill-Qwen-7B_W4A16_RK3576.rkllm
COPY ./src/flask_server.py /app/
EXPOSE 8080
CMD ["python", "/app/flask_server.py", "--rkllm_model_path", "/app/models/DeepSeek-R1-Distill-Qwen-7B_W4A16_RK3576.rkllm", "--target_platform", "rk3576"]

# 4. DeepSeek-R1-Distill-Qwen-7B_W4A16_G128 (修正了拼写 Owen -> Qwen)
FROM base AS deepseek-r1-distill-qwen-7b-w4a16-g128-rk3576
RUN wget https://huggingface.co/JiahaoLi/DeepSeek-R1-Distill-Qwen-RK3576/resolve/main/DeepSeek-R1-Distill-Qwen-7B_W4A16_G128_RK3576.rkllm -o /app/models/DeepSeek-R1-Distill-Qwen-7B_W4A16_G128_RK3576.rkllm
COPY ./src/flask_server.py /app/
EXPOSE 8080
CMD ["python", "/app/flask_server.py", "--rkllm_model_path", "/app/models/DeepSeek-R1-Distill-Qwen-7B_W4A16_G128_RK3576.rkllm", "--target_platform", "rk3576"]