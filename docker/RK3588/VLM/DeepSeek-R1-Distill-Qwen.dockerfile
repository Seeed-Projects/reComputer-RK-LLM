# 第一阶段：环境准备
FROM python:3.10-slim AS base

RUN apt-get update && \
    apt-get install -y wget curl git sudo libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/models

# 安装依赖
COPY ./src/vlm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝运行时库和工具
COPY ./lib/librkllmrt.so /usr/lib/librkllmrt.so 
COPY ./lib/librkllm_service.so /usr/lib/librkllm_service.so 
COPY ./src/fix_freq_rk3588.sh /app/fix_freq_rk3588.sh
RUN chmod +x /app/fix_freq_rk3588.sh

# --- 最终阶段 ---
FROM base AS final
ARG LLM_URL
ARG LLM_MODEL
ARG VISION_URL
ARG VISION_MODEL

# 必须在这里重新下载或声明，因为 ARG 在不同阶段不共享
RUN if [ -n "${LLM_URL}" ]; then \
    wget --progress=dot:giga "${LLM_URL}}" -O "/app/models/${LLM_MODEL}"; \
    fi
RUN if [ -n "${VISION_URL}" ]; then \
    wget --progress=dot:giga "${LLM_URL}}" -O "/app/models/${VISION_MODEL}"; \
    fi

COPY ./src/vlm/fastapi_server_vlm.py /app/

# 将 ARG 转为 ENV，这样 CMD 才能读取到
ENV LLM_MODEL_PATH=/app/models/${LLM_MODEL}
ENV VISION_MODEL_PATH=/app/models/$P{VISION_MODEL}


EXPOSE 8002

CMD ["sh", "-c", "python3 /app/fastapi_server_vlm.py --llm_model ${LLM_MODEL_PATH} --encoder_model ${VISION_MODEL_PATH}"]