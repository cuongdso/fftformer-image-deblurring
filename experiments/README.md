# Quản lý thực nghiệm

Mỗi experiment trong tương lai tạo thư mục `experiments/<experiment-id>/` với `metadata.md` dựa trên schema dưới đây; không commit dataset, checkpoints, ảnh output hoặc log lớn.

```text
Experiment ID:
Group commit SHA:
Upstream baseline SHA: 01306b5d4072146a011c8c6e9d4927cf4607f1d5
Model/checkpoint (source + checksum if available):
Dataset (source + version/license):
Configuration:
Python / PyTorch / CUDA:
Hardware:
Random seed:
Evaluation protocol:
Metrics (measured only):
Runtime/memory (measured only):
Artifacts stored outside Git:
Failure cases / limitations:
```

Không ghi PSNR/SSIM, runtime hay memory khi chưa đo bằng protocol được ghi rõ.
