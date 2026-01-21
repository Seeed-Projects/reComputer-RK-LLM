# 第一阶段：环境准备
FROM python:3.10-slim AS base

RUN apt-get update && \
    apt-get install -y wget curl git sudo libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/models

# 安装依赖
COPY ./src/fastapi_server_requirements_llm.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝运行时库和工具
COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh

# --- 最终阶段 ---
FROM base AS final
ARG MODEL_URL
ARG MODEL_FILE

# 必须在这里重新下载或声明，因为 ARG 在不同阶段不共享
RUN if [ -n "${MODEL_URL}" ]; then \
    wget --progress=dot:giga "${MODEL_URL}" -O "/app/models/${MODEL_FILE}"; \
    fi

COPY ./src/fastapi_server_llm.py /app/

# 将 ARG 转为 ENV，这样 CMD 才能读取到
ENV RKLLM_MODEL_PATH=/app/models/${MODEL_FILE}

EXPOSE 8080

CMD ["sh", "-c", "python3 /app/fastapi_server_llm.py --rkllm_model_path ${RKLLM_MODEL_PATH} --target_platform rk3576"]