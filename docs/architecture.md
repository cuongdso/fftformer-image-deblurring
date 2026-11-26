# Kiến trúc repository

## Quyết định nền tảng

Repository tách **official baseline** khỏi **group implementation**. `third_party/FFTformer/` là submodule chỉ đọc theo quy ước, pin tại `01306b5d4072146a011c8c6e9d4927cf4607f1d5`. `src/fftformer_lab/` là không gian duy nhất cho mã nhóm. Lý do và cách cập nhật pin nằm ở [ADR-0001](decisions/0001-upstream-baseline.md).

```text
Ảnh / manifest ─► WS2 data ────────► WS3 inference ─► ảnh + metadata
Checkpoint + kiến trúc upstream ──► WS1 ──► WS3

WS2 Sample (GT) + WS3 InferenceResult ─► WS4 evaluation API
Thư mục prediction + target ─► WS4 evaluation CLI ─► PSNR/SSIM + metadata
```

## Thành phần chính

| Thành phần | Trách nhiệm | Có trên `main` |
| --- | --- | --- |
| `third_party/FFTformer/` | Mã upstream để đối chiếu/reproduce | Git submodule tại SHA đã pin |
| `src/fftformer_lab/model/` | Model adapter | `ModelSpec`, `ModelAdapter`, strict `load_model` |
| `src/fftformer_lab/data/` | Đọc/chuẩn hóa ảnh RGB và ghép cặp | Paired/single manifest và `ImageDataset` |
| `src/fftformer_lab/inference/` | Điều phối khử mờ | Pipeline pad/crop, lưu PNG/JPEG, CLI ảnh đơn/thư mục/manifest |
| `src/fftformer_lab/evaluation/` | Đánh giá theo protocol và lưu metadata | PSNR/SSIM CPU, kiểm tra config và CLI đánh giá prediction có sẵn |
| `docs/contracts.md` | Hợp đồng giữa module | `APPROVED` baseline v1.0; xem ADR-0002 |

Không module nào được sửa hoặc giả định interface của module khác trước khi contract liên quan được Tech Lead chốt trong ADR/contract. Mọi thay đổi sau đó cần scope Tech Lead cho phép.

## Layout và quyền sở hữu

`docs/ownership.md` là nguồn chuẩn về 4 workstream. Các module có thể phụ thuộc theo chiều `model/data → inference → evaluation`, nhưng WS4 chỉ điều phối interface; không sửa nghiệp vụ chuyên môn của WS1–WS3 mà không thống nhất.

## Hướng phát triển có kiểm soát

- Baseline giữ nguyên để so sánh; đổi SHA submodule là thay đổi có review và ADR/audit cập nhật.
- Experiment hoặc cải tiến kiến trúc ở package/config riêng, không ghi đè hành vi baseline.
- Config, artifact, metric và environment của kết quả nhóm phải truy vết bằng schema trong `experiments/README.md`.
- `src/fftformer_lab/app/` dành cho demo UI `PLANNED`.
