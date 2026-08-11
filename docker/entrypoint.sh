#!/bin/sh
set -eu

detect_platform() {
    if [ "${TARGET_PLATFORM:-auto}" != "auto" ]; then
        printf '%s\n' "$TARGET_PLATFORM"
        return
    fi

    compatible=""
    if [ -r /proc/device-tree/compatible ]; then
        compatible="$(tr '\000' '\n' < /proc/device-tree/compatible 2>/dev/null || true)"
    fi

    case "$compatible" in
        *rk3576*) printf '%s\n' rk3576 ;;
        *rk3588s*|*rk3588*) printf '%s\n' rk3588 ;;
        *)
            echo "TARGET_PLATFORM=auto could not detect RK3576/RK3588; set TARGET_PLATFORM explicitly." >&2
            exit 1
            ;;
    esac
}

TARGET_PLATFORM="$(detect_platform)"
export TARGET_PLATFORM

if [ "${RUN_FREQ_FIX:-true}" = "true" ]; then
    case "$TARGET_PLATFORM" in
        rk3576) bash /app/fix_freq_rk3576.sh || echo "Warning: RK3576 frequency setup failed; continuing." >&2 ;;
        rk3588|rk3588s) bash /app/fix_freq_rk3588.sh || echo "Warning: RK3588 frequency setup failed; continuing." >&2 ;;
        *) echo "Unsupported frequency profile for $TARGET_PLATFORM; continuing." >&2 ;;
    esac
fi

if [ "${1:-serve}" != "serve" ]; then
    exec "$@"
fi
shift || true

if [ -n "${MODEL_FILE:-}" ]; then
    case "$MODEL_FILE" in
        /*) MODEL_PATH="$MODEL_FILE" ;;
        *) MODEL_PATH="/app/models/$MODEL_FILE" ;;
    esac
fi
if [ -n "${VISION_MODEL_FILE:-}" ]; then
    case "$VISION_MODEL_FILE" in
        /*) VISION_MODEL_PATH="$VISION_MODEL_FILE" ;;
        *) VISION_MODEL_PATH="/app/models/$VISION_MODEL_FILE" ;;
    esac
fi
export MODEL_PATH VISION_MODEL_PATH

if [ ! -f "${MODEL_PATH}" ]; then
    echo "RKLLM model not found: ${MODEL_PATH}" >&2
    echo "Mount a model there or use a model image built from models/." >&2
    exit 1
fi

case "${MODEL_KIND:-llm}" in
    llm)
        if [ "${INTERACTIVE_CHAT:-false}" != "true" ]; then
            set -- --no_chat "$@"
        fi
        exec python3 /app/fastapi_server_llm.py \
            --rkllm_model_path "${MODEL_PATH}" \
            --target_platform "${TARGET_PLATFORM}" \
            --port "${PORT:-8001}" \
            --model_name "${API_MODEL_NAME:-rkllm-model}" \
            "$@"
        ;;
    vlm)
        if [ ! -f "${VISION_MODEL_PATH:-}" ]; then
            echo "VLM vision model not found: ${VISION_MODEL_PATH:-}" >&2
            exit 1
        fi
        case "$TARGET_PLATFORM" in
            rk3576) rknn_core_num=2 ;;
            rk3588|rk3588s) rknn_core_num=3 ;;
            *) rknn_core_num=3 ;;
        esac
        exec python3 /app/fastapi_server_vlm.py \
            --encoder_model "${VISION_MODEL_PATH}" \
            --llm_model "${MODEL_PATH}" \
            --target_platform "${TARGET_PLATFORM}" \
            --port "${PORT:-8001}" \
            --rknn_core_num "$rknn_core_num" \
            --img_start "${VLM_IMAGE_START:-<|vision_start|>}" \
            --img_end "${VLM_IMAGE_END:-<|vision_end|>}" \
            --img_content "${VLM_IMAGE_CONTENT:-<|image_pad|>}" \
            "$@"
        ;;
    *)
        echo "Unsupported MODEL_KIND: ${MODEL_KIND:-} (expected llm or vlm)" >&2
        exit 1
        ;;
esac
