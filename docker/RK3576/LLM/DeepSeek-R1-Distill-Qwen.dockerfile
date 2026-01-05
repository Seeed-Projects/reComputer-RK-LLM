# --- 通用基础阶段 ---
FROM python:3.10-slim AS base

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y wget curl git sudo && \
    rm -rf /var/lib/apt/lists/*

# 创建目录
RUN mkdir -p /app/models

# 安装Python依赖
COPY ./src/flask_server_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复制 RKLLM 运行时库
COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh

# --- 模型构建阶段 (使用 ARG 动态构建) ---
FROM base AS model-image

# 接收来自 workflow 的参数
ARG MODEL_URL
ARG MODEL_FILE

# 下载模型并重命名
# 注意：wget -O (大写O) 用于指定输出文件名
RUN wget --progress=dot:giga "${MODEL_URL}" -O "/app/models/${MODEL_FILE}"

# 复制服务器代码
COPY ./src/flask_server.py /app/

# 设置环境变量，以便 CMD 使用
ENV MODEL_PATH=/app/models/${MODEL_FILE}

EXPOSE 8080

# 启动命令：使用环境变量引用正确的文件
CMD python /app/flask_server.py --rkllm_model_path ${MODEL_PATH} --target_platform rk3576