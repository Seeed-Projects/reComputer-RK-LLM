# ==========================================
# 阶段 1: 通用基础环境
# ==========================================
FROM python:3.10-slim AS base

# 安装系统基础依赖
RUN apt-get update && \
    apt-get install -y wget curl git sudo && \
    rm -rf /var/lib/apt/lists/*

# 创建模型存放目录
RUN mkdir -p /app/models

# 安装 Python 依赖（利用缓存，只有 requirements 变化时才重新安装）
COPY ./src/flask_server_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复制 RKLLM 驱动库和硬件优化脚本
COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh

# ==========================================
# 阶段 2: 动态模型构建 (Matrix 目标阶段)
# ==========================================
FROM base AS model-image

# 接收来自 Workflow Matrix 的参数
ARG MODEL_URL
ARG MODEL_FILE

# 从 Hugging Face 下载模型并重命名保存
# 使用 -O (大写) 确保下载到指定路径和文件名
RUN wget --progress=dot:giga "${MODEL_URL}" -O "/app/models/${MODEL_FILE}"

# 复制 Flask 服务代码
COPY ./src/flask_server.py /app/

# 设置模型路径环境变量，供 CMD 调用
ENV RKLLM_MODEL_PATH=/app/models/${MODEL_FILE}

EXPOSE 8080

# 启动命令
CMD python /app/flask_server.py --rkllm_model_path ${RKLLM_MODEL_PATH} --target_platform rk3576