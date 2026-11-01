# inference-from-scratch

LLM inference engine internals, built one piece at a time from first principles: attention, KV cache, paged memory, continuous batching, sampling, speculative decoding, quantization, and CUDA kernels.

Pure numpy for the engine; PyTorch is used only as a reference in tests.

```bash
pip install numpy torch pytest
python -m pytest -q
```

## What's inside

- `infer/ops.py` — numerically stable softmax, RMSNorm
