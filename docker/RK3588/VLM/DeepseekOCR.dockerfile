# 第一阶段：环境准备
FROM python:3.10-slim AS base

RUN apt-get update && \
    apt-get install -y wget curl git sudo libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/models

# 安装依赖
COPY ./src/vlm/requirements.txt /app/requirements.txt
COPY ./src/vlm/fastapi_server_vlm.py /app/fastapi_server_vlm.py
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝运行时库和工具
COPY ./lib/librkllmrt.so /usr/lib/librkllmrt.so 
COPY ./lib/librkllm_service.so /usr/lib/librkllm_service.so 
COPY ./lib/librknnrt.so /usr/lib/librknnrt.so
  

COPY ./src/fix_freq_rk3588.sh /app/fix_freq_rk3588.sh
RUN chmod +x /app/fix_freq_rk3588.sh

# --- 最终阶段 ---
FROM base AS final
ARG LLM_URL
ARG LLM_FILE           # 改为LLM_FILE
ARG VISION_URL
ARG VISION_FILE        # 改为VISION_FILE

# 检查变量是否设置，如果设置了就下载模型
RUN if [ -n "${LLM_URL}" ] && [ -n "${LLM_FILE}" ]; then \
    wget --progress=dot:giga "${LLM_URL}" -O "/app/models/${LLM_FILE}"; \
    else \
    echo "LLM_URL or LLM_FILE not set, skipping LLM model download"; \
    fi

RUN if [ -n "${VISION_URL}" ] && [ -n "${VISION_FILE}" ]; then \
    wget --progress=dot:giga "${VISION_URL}" -O "/app/models/${VISION_FILE}"; \
    else \
    echo "VISION_URL or VISION_FILE not set, skipping vision model download"; \
    fi

# 将 ARG 转为 ENV，这样 CMD 才能读取到
ENV LLM_MODEL_PATH=/app/models/${LLM_FILE}
ENV VISION_MODEL_PATH=/app/models/${VISION_FILE}

EXPOSE 8002

CMD ["sh", "-c", "python3 /app/fastapi_server_vlm.py --llm_model ${LLM_MODEL_PATH} --encoder_model ${VISION_MODEL_PATH}"]