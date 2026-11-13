# Kiến trúc repository

## Quyết định nền tảng

Repository tách **official baseline** khỏi **group implementation**. `third_party/FFTformer/` là submodule chỉ đọc theo quy ước, pin tại `01306b5d4072146a011c8c6e9d4927cf4607f1d5`. `src/fftformer_lab/` là không gian duy nhất cho mã nhóm. Lý do và cách cập nhật pin nằm ở [ADR-0001](decisions/0001-upstream-baseline.md).

```text
external image / paired dataset / checkpoint
                 │
                 ▼
       WS2 data contract ──► WS3 inference contract ──► output image/result
                 │                      │
                 ▼                      ▼
       WS1 model-loading contract ─► WS4 evaluation contract ─► experiment metadata
                 │
                 └── compares with pinned upstream baseline (no direct edits)
```

## Thành phần chính

| Thành phần | Trách nhiệm | Trạng thái M0 |
| --- | --- | --- |
| `third_party/FFTformer/` | Mã upstream để đối chiếu/reproduce | pinned submodule, không chạy ở M0 |
| `src/fftformer_lab/model/` | Adapter/analysis model của nhóm | ranh giới tài liệu |
| `src/fftformer_lab/data/` | Dataset, validation và preprocessing | ranh giới tài liệu |
| `src/fftformer_lab/inference/` + `app/` | CLI/wrapper/demo về sau | ranh giới tài liệu |
| `src/fftformer_lab/evaluation/` | Metric/protocol/metadata integration | ranh giới tài liệu |
| `docs/contracts.md` | Hợp đồng giữa module | `APPROVED` baseline v1.0; xem ADR-0002 |

Không module nào được sửa hoặc giả định interface của module khác trước khi contract liên quan được Tech Lead chốt trong ADR/contract. Mọi thay đổi sau đó cần scope Tech Lead cho phép.

## Layout và quyền sở hữu

`docs/ownership.md` là nguồn chuẩn về 4 workstream. Các module có thể phụ thuộc theo chiều `model/data → inference → evaluation`, nhưng WS4 chỉ điều phối interface; không sửa nghiệp vụ chuyên môn của WS1–WS3 mà không thống nhất.

## Hướng phát triển có kiểm soát

- Baseline giữ nguyên để so sánh; đổi SHA submodule là thay đổi có review và ADR/audit cập nhật.
- Experiment hoặc cải tiến kiến trúc ở package/config riêng, không ghi đè hành vi baseline.
- Config, artifact, metric và environment của kết quả nhóm phải truy vết bằng schema trong `experiments/README.md`.
- Không có implementation production ở M0; directory README là boundary có chủ đích, không phải bằng chứng chức năng đã hoàn thành.
