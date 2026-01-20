# --- 第一阶段：构建基础环境 ---
FROM python:3.10-slim AS builder

# 安装系统依赖
# libgomp1 是多线程推理必须的，ca-certificates 确保 wget 正常工作
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget curl git libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖
# 建议将 flask 换成 fastapi + uvicorn，如果脚本没改，这里保持原样
COPY ./src/flask_server_requirements_llm.txt .
RUN pip install --no-cache-dir -r flask_server_requirements_llm.txt

# --- 第二阶段：运行时镜像 ---
FROM builder AS final

WORKDIR /app

# 拷贝 RKLLM 核心运行时库 (这是 NPU 推理的关键)
COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so
# 自动同步库缓存
RUN ldconfig

# 拷贝脚本与工具
COPY ./src/fix_freq_rk3588.sh /app/
COPY ./src/flask_server_llm.py /app/main.py
RUN chmod +x /app/fix_freq_rk3588.sh

# 处理模型文件
ARG MODEL_URL
ARG MODEL_FILE=model.rknn
RUN mkdir -p /app/models
RUN if [ -z "${MODEL_URL}" ]; then \
        echo "Warning: MODEL_URL not set, ensure model is mounted via volume"; \
    else \
        wget --progress=dot:giga "${MODEL_URL}" -O "/app/models/${MODEL_FILE}"; \
    fi

# 环境变量设置
ENV RKLLM_MODEL_PATH=/app/models/${MODEL_FILE}
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 8080

# 启动命令：增加自动频率锁定（针对 RK3588 优化性能）
CMD ["sh", "-c", "/app/fix_freq_rk3588.sh && python3 /app/main.py --rkllm_model_path ${RKLLM_MODEL_PATH} --target_platform rk3588"]