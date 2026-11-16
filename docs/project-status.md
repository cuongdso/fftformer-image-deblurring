# Trạng thái dự án

## M0 — Repository Foundation

| Hạng mục | Trạng thái | Evidence |
| --- | --- | --- |
| Baseline upstream pin | DONE | submodule `third_party/FFTformer` tại `01306b5…` |
| Audit code/source | DONE (static audit) | `docs/upstream-audit.md` |
| Team/agent onboarding | DONE | `AGENTS.md`, `docs/onboarding.md` |
| Workstream, ownership, contracts | DONE (M0 candidate `PROPOSED`; được M1.0 supersede bởi v1.0 `APPROVED`) | `docs/ownership.md`, `docs/contracts.md`, `docs/decisions/0002-m1-contract-baseline.md` |
| Tutorial reproduction | DONE (audit-derived, chưa chạy) | `docs/reproduction/` |
| Tooling không GPU | DONE | `scripts/check_env.py`, unit test |
| CI lightweight | CONFIGURED (M0); run success sau transfer | `.github/workflows/ci.yml`; Actions run `43360793495` (commit `fd918794`) |
| Baseline inference | NOT RUN | không tải/chạy checkpoint ở M0 |
| Benchmark/reproduction | NOT RUN | không có dataset hoặc metric group |
| Training/fine-tuning/demo | NOT STARTED | ngoài phạm vi M0 |

`DONE` mô tả artifact M0 đã tồn tại, không đồng nghĩa inference/benchmark đã thành công. Chỉ cập nhật bảng này cùng evidence (command, log, PR/commit) khi trạng thái thay đổi.

## M1.0 — Contract alignment

- **Trạng thái:** `APPROVED` — contract baseline v1.0 do Tech Lead chốt; M1.0 documentation finalization complete.
- **Evidence:** [contracts](contracts.md), [M1.0 decision](m1-interface-review.md), [ADR-0002](decisions/0002-m1-contract-baseline.md) và [M1 backlog](backlog.md).
- **Scope:** chốt ranh giới model/data/inference/evaluation và giải vòng dependency. Không có inference/training/benchmark hoặc implementation nào đã chạy.

## M1 — WS1-01 architecture/checkpoint audit

- **Trạng thái:** báo cáo đã merge tại [PR #4](https://github.com/cuongdso/fftformer-image-deblurring/pull/4), commit `237cbf7`; Owner `DucHai27`.
- **Evidence cập nhật:** chạy lại `python -B scripts/ws1_01_verify.py --repo . --output docs/research/ws1-01-runtime-evidence.json` — 10/10 PASS trên CPU. JSON được tái tạo để SHA-256 của script khớp nội dung đã version; dùng LF khi ghi file.
- **Phạm vi:** mapping paper/code, audit constructor/config/checkpoint và shape/compatibility test plan tại [báo cáo](research/fftformer-architecture.md). CPU synthetic smoke không phải benchmark chất lượng.

## M1 — WS1-02 model specification/adapter

- **Trạng thái:** bản triển khai trên working branch, chờ review; chưa đánh dấu task `DONE`.
- **Đã có:** `ModelConfig`, `ModelSpec`, `ModelMetadata`, `ModelAdapter`, `build_model` trong `src/fftformer_lab/model/`; khởi tạo upstream tường minh, model `eval()`, validation C-02 và metadata model chưa nạp trọng số.
- **Evidence sau sửa lỗi dtype/device ngày 16/11/2026:** `python -B -m unittest discover -s tests -p "test_*.py"` chạy 14 test thành công, skip 16; bật `FFTFORMER_RUN_MODEL_TESTS=1` và chạy cùng lệnh thêm `-v`: toàn bộ 30 test thành công trên CPU, gồm 4 test hồi quy default dtype/device của caller. Chi tiết môi trường, scope và test source tại [model guide](modules/model.md#kiểm-thử-đã-chạy).
- **Giới hạn:** chưa có checkpoint loader WS1-03, chưa chạy checkpoint qua adapter, chưa xác minh CUDA forward hoặc benchmark. Không suy ra baseline reproduction hay pipeline WS2/WS3/WS4 đã hoàn tất.

## Milestone đề xuất

- `M1` — Official Baseline Reproduction
- `M2` — Dataset & Evaluation Pipeline
- `M3` — Model Analysis & Inference Integration
- `M4` — Experiments & Demonstration
- `M5` — Final Validation & Academic Deliverables

Các milestone trên là `PLANNED`, chưa được đánh dấu hoàn thành.
