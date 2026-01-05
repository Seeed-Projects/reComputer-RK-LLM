# 使用 TARGETPLATFORM 确保多架构构建稳定性
FROM --platform=$TARGETPLATFORM python:3.10-slim AS base

# 核心修复：安装 libgomp1 解决运行时 OSError，安装 file 用于校验
RUN apt-get update && \
    apt-get install -y wget curl git sudo libgomp1 file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models

# 安装 Python 依赖（Flask, numpy 等）
COPY ./src/flask_server_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝 RKLLM 运行时环境
COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh

# --- 关键校验：防止 Git LFS 导致的空文件或指针文件进入镜像 ---
RUN if ! head -c 4 /usr/lib/librkllmrt.so | grep -q $'\x7fELF'; then \
        echo "ERROR: librkllmrt.so is corrupted or a Git LFS pointer! Please check 'lfs: true' in workflow."; \
        exit 1; \
    fi

# --- 模型集成阶段 ---
FROM base AS model-image
ARG MODEL_URL
ARG MODEL_FILE

# 下载对应的模型文件
RUN wget --progress=dot:giga "${MODEL_URL}" -O "/app/models/${MODEL_FILE}"

COPY ./src/flask_server.py /app/
ENV RKLLM_MODEL_PATH=/app/models/${MODEL_FILE}
EXPOSE 8080

# 启动推理服务器
CMD ["sh", "-c", "python /app/flask_server.py --rkllm_model_path ${RKLLM_MODEL_PATH} --target_platform rk3576"]