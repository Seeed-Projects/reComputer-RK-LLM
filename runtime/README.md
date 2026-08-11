# Runtime artifacts

This directory is the single source of truth for the ARM64 runtime packaged
into the Docker environment image.

- `librkllmrt.so`: RKLLM runtime v1.3.0
- `librknnrt.so`: RKNN runtime used by the VLM encoder
- `wheels/`: ARM64 Python runtime wheels

The runtime version must match the ctypes ABI used by the application. Update
the binary and `RKLLM_RUNTIME_VERSION` together, then rebuild the environment
image before publishing model images.
