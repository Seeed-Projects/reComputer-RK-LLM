# 基础阶段
FROM --platform=$TARGETPLATFORM python:3.10-slim AS base

# 安装 libgomp1 (修复 OpenMP 报错) 和 file (用于校验)
RUN apt-get update && \
    apt-get install -y wget curl git sudo libgomp1 file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models

# 安装依赖
COPY ./src/flask_server_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝运行时库
COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh

# --- 【关键防错】检查 LFS 文件是否正确加载 ---
RUN if ! head -c 4 /usr/lib/librkllmrt.so | grep -q $'\x7fELF'; then \
        echo "ERROR: librkllmrt.so is not a valid ELF file! Check Git LFS."; \
        exit 1; \
    fi

# 模型集成阶段
FROM base AS model-image
ARG MODEL_URL
ARG MODEL_FILE

RUN wget --progress=dot:giga "${MODEL_URL}" -O "/app/models/${MODEL_FILE}"

COPY ./src/flask_server.py /app/
ENV RKLLM_MODEL_PATH=/app/models/${MODEL_FILE}
EXPOSE 8080

CMD ["sh", "-c", "python /app/flask_server.py --rkllm_model_path ${RKLLM_MODEL_PATH} --target_platform rk3576"]