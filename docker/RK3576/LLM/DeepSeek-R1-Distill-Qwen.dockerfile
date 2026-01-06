# 使用基础镜像
FROM python:3.10-slim AS base

# 安装必要的系统库，解决 libgomp.so.1 缺失问题
RUN apt-get update && \
    apt-get install -y wget curl git sudo libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models

# 安装 Python 环境依赖
COPY ./src/flask_server_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 拷贝运行时库（已取消之前的校验逻辑）
COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh

# 模型集成阶段
FROM base AS model-image
ARG MODEL_URL
ARG MODEL_FILE

# 从网络下载模型，避免 Docker 上传压力
RUN wget --progress=dot:giga "${MODEL_URL}" -O "/app/models/${MODEL_FILE}"

COPY ./src/flask_server.py /app/
ENV RKLLM_MODEL_PATH=/app/models/${MODEL_FILE}
EXPOSE 8080

CMD ["sh", "-c", "python /app/flask_server.py --rkllm_model_path ${RKLLM_MODEL_PATH} --target_platform rk3576"]