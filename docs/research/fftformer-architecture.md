# FFTformer — mapping paper ↔ code

**Evidence:** static inspection `third_party/FFTformer/.../fftformer_arch.py` tại SHA pin; paper là nguồn chuẩn cho diễn giải phương pháp.

- Backbone: overlap 3×3 embedding → 3 encoder levels → decoder levels + fusion → refinement → residual output.
- Encoder blocks trong code dùng DFFN với `att=False`; decoder/refinement set `att=True`, kích hoạt FSAS. Điều này phù hợp mô tả README rằng FSAS chỉ dùng decoder.
- `FSAS`: Q/K/V từ point-wise + depth-wise conv; split patch 8; `rfft2`, Q×K element-wise, `irfft2`, normalize, multiply V, project.
- `DFFN`: expand channels, patch-FFT with learned frequency parameter, inverse FFT, depth-wise conv, GELU gate and projection.

Các hyperparameter/config/checkpoint correspondence vẫn `UNVERIFIED` cho đến WS1. Không dùng mapping này làm lý do đổi architecture.
