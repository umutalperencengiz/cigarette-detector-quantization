# Model artifacts

Only the graphs the live demo loads are committed to this repository:

| File | Size | Used by |
| --- | --- | --- |
| `yolo11n_int8_static_640.onnx` | 4.4 MB | demo default, `results/int8_static_640.json` |
| `yolo11n_fp32_640.onnx` | 10.6 MB | demo A/B baseline, `results/fp32_640.json` |
| `yolo11n_int8_static_320.onnx` | 4.3 MB | demo fast option (`--size 320`) |
| `yolo11n_fp32_320.onnx` | 10.0 MB | demo fast option (`--size 320`) |

Everything else in the study is generated, not stored:

```bash
python experiments/export_variants.py
```

That script needs the original Ultralytics checkpoint at
`models/yolo11n_fp32_640.pt` (the `train15` run of the companion study) and
produces:

- `yolo11n_fp32_{640,512,416,320}.onnx` — the input-size sweep
- `yolo11n_fp16_640.onnx` — float16 weights, float32 I/O
- `yolo11n_int8_dynamic_640.onnx` — dynamic weight quantization
- `yolo11n_int8_static_640.onnx` — QDQ, calibrated on 256 training frames

All exports use opset 13. Opset 12 produced QDQ models that ONNX Runtime
rejects with `Unrecognized attribute: axis for operator DequantizeLinear`.

The 640 graphs take a `[1, 3, 640, 640]` float32 RGB tensor in `0..1` and return
`[1, 5, 8400]` (four box channels plus one class score) in letterboxed pixel
space.
