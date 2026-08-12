# Runtime artifacts

This directory is the single source of truth for the ARM64 runtime packaged
into the Docker environment image.

- `librkllmrt.so`: RKLLM runtime v1.3.0
- `librknnrt.so`: RKNN runtime used by the VLM encoder
- `wheels/`: ARM64 Python runtime wheels

`librkllmrt.so` is copied from the official `release-v1.3.0` ARM64 Linux
runtime at `airockchip/rknn-llm` and has SHA-256
`6a9e4fc5324c68921c3a900340361e107af7599fe34dc8fa7759b2c5ae22a6e6`.

The runtime version must match the ctypes ABI used by the application. Update
the binary and `RKLLM_RUNTIME_VERSION` together, then rebuild the environment
image before publishing model images.
