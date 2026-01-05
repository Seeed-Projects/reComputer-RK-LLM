# 多阶段构建 - 为每个模型创建单独的镜像

# 通用基础阶段
FROM python:3.10-slim as base

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y wget curl git && \
    rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir rkllm

# 创建模型目录
RUN mkdir -p /app/models

# 下载RKLLM运行时库
RUN wget https://github.com/airockchip/rknn-llm/releases/download/v1.0.0/librkllmrt.so -O /usr/lib/librkllmrt.so

# 安装RK3576频率修复脚本
RUN wget https://github.com/airockchip/rknn-llm/raw/main/scripts/fix_freq_rk3576.sh -O /app/fix_freq_rk3576.sh && \
    chmod +x /app/fix_freq_rk3576.sh

# 运行频率修复脚本
RUN /app/fix_freq_rk3576.sh


# DeepSeek-R1-Distill-Qwen-1.5B_FP16_RK3576 镜像
FROM base as deepseek-r1-distill-qwen-1.5b-fp16-rk3576
COPY ./models/RK3576/LLM/DeepSeek-R1-Distill-Qwen/DeepSeek-R1-Distill-Qwen-1.5B_FP16_RK3576.sh /app/scripts/
RUN chmod +x /app/scripts/DeepSeek-R1-Distill-Qwen-1.5B_FP16_RK3576.sh
RUN cd /app/scripts && ./DeepSeek-R1-Distill-Qwen-1.5B_FP16_RK3576.sh
RUN mv /app/scripts/DeepSeek-R1-Distill-Qwen-1.5B_FP16_RK3576.rkllm /app/models/
COPY ./src/llm_api.py /app/
EXPOSE 8000
CMD ["python", "/app/llm_api.py"]


# DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3576 镜像
FROM base as deepseek-r1-distill-qwen-1.5b-w4a16-rk3576
COPY ./models/RK3576/LLM/DeepSeek-R1-Distill-Qwen/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3576.sh /app/scripts/
RUN chmod +x /app/scripts/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3576.sh
RUN cd /app/scripts && ./DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3576.sh
RUN mv /app/scripts/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3576.rkllm /app/models/
COPY ./src/llm_api.py /app/
EXPOSE 8000
CMD ["python", "/app/llm_api.py"]


# DeepSeek-R1-Distill-Qwen-7B_W4A16_RK3576 镜像
FROM base as deepseek-r1-distill-qwen-7b-w4a16-rk3576
COPY ./models/RK3576/LLM/DeepSeek-R1-Distill-Qwen/DeepSeek-R1-Distill-Qwen-7B_W4A16_RK3576.sh /app/scripts/
RUN chmod +x /app/scripts/DeepSeek-R1-Distill-Qwen-7B_W4A16_RK3576.sh
RUN cd /app/scripts && ./DeepSeek-R1-Distill-Qwen-7B_W4A16_RK3576.sh
RUN mv /app/scripts/DeepSeek-R1-Distill-Qwen-7B_W4A16_RK3576.rkllm /app/models/
COPY ./src/llm_api.py /app/
EXPOSE 8000
CMD ["python", "/app/llm_api.py"]


# DeepSeek-R1-Distill-Owen-7B_W4A16_G128_RK3576 镜像
FROM base as deepseek-r1-distill-owen-7b-w4a16-g128-rk3576
COPY ./models/RK3576/LLM/DeepSeek-R1-Distill-Qwen/DeepSeek-R1-Distill-Owen-7B_W4A16_G128_RK3576.sh /app/scripts/
RUN chmod +x /app/scripts/DeepSeek-R1-Distill-Owen-7B_W4A16_G128_RK3576.sh
RUN cd /app/scripts && ./DeepSeek-R1-Distill-Owen-7B_W4A16_G128_RK3576.sh
RUN mv /app/scripts/DeepSeek-R1-Distill-Qwen-7B_W4A16_G128_RK3576.rkllm /app/models/DeepSeek-R1-Distill-Owen-7B_W4A16_G128_RK3576.rkllm
COPY ./src/llm_api.py /app/
EXPOSE 8000
CMD ["python", "/app/llm_api.py"]