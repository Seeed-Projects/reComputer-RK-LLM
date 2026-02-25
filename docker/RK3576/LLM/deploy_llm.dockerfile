# 第一阶段：环境准备
FROM python:3.10-slim AS base

RUN apt-get update && \
    apt-get install -y wget curl git sudo libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/models

# 安装依赖
COPY ./src/llm/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝运行时库和工具
COPY ./lib/librkllmrt.so /usr/lib/librkllmrt.so
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh
RUN bash /app/fix_freq_rk3576.sh

# --- 最终阶段 ---
FROM base AS final

COPY ./src/llm/fastapi_server_llm.py /app/

EXPOSE 8001

