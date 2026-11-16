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
- **Evidence tại bản WS1-02 sau sửa lỗi dtype/device ngày 16/11/2026:** `python -B -m unittest discover -s tests -p "test_*.py"` chạy 14 test thành công, skip 16; bật `FFTFORMER_RUN_MODEL_TESTS=1` và chạy cùng lệnh thêm `-v`: toàn bộ 30 test thành công trên CPU, gồm 4 test hồi quy default dtype/device của caller. Chi tiết môi trường, scope và test source tại [model guide](modules/model.md#kiểm-thử-ws1-02-trước-khi-bổ-sung-loader).
- **Giới hạn bản WS1-02:** chưa có checkpoint loader, chưa chạy checkpoint qua adapter; phần đó được bổ sung riêng trong WS1-03 bên dưới. Không suy ra baseline reproduction hay pipeline WS2/WS3/WS4 đã hoàn tất.

## M1 — WS1-03 strict checkpoint loader

- **Trạng thái:** bản triển khai trên feature branch dựa trên WS1-02, chờ review; chưa đánh dấu `DONE`.
- **Đã có:** `load_model` với raw tensor state-dict schema validation, key/shape/dtype checks, `weights_only=True`, `map_location` theo device đã resolve và strict load; checkpoint identity ghi trong metadata của adapter ở `eval()`.
- **Evidence sau sửa lỗi review:** `python -B scripts/ws1_03_verify.py --repo . --output docs/research/ws1-03-runtime-evidence.json` — 55/55 test thành công trên CPU, gồm hồi quy dtype/device, GoPro thật dưới caller defaults khác và 8 test reporter/subtest, bao gồm subtest trùng nhãn. [JSON](research/ws1-03-runtime-evidence.json) ghi từng test và metadata/hash thực tế. Chạy `python -B -S -m unittest discover -s tests -p "test_*.py"`: 22 test qua, 33 test opt-in được skip.
- **Artifact:** GoPro checkpoint sẵn có trong upstream submodule, SHA-256 `336717b7c4e97b66af112cc7878106cb185b698bd872c2a8247911507540b2a3`; strict load + forward hai shape qua API nhóm. Fixture khác là Conv2d mock tự tạo, không phải weights FFTformer. Không thêm dataset/checkpoint vào Git nhóm.
- **Giới hạn:** CUDA forward, RealBlur checkpoint, benchmark chất lượng và training chưa xác minh/chưa chạy. Không thay đổi contracts hoặc upstream source/pin.
- **Bàn giao:** Owner `DucHai27`; [hồ sơ WS1-01/02/03](research/ws1-handoff.md) có files, API, commands/results, commit, trạng thái PR/reviewer và dependency tiếp theo. Chờ WS3/Tech Lead review và CI của PR.

## Milestone đề xuất

- `M1` — Official Baseline Reproduction
- `M2` — Dataset & Evaluation Pipeline
- `M3` — Model Analysis & Inference Integration
- `M4` — Experiments & Demonstration
- `M5` — Final Validation & Academic Deliverables

Các milestone trên là `PLANNED`, chưa được đánh dấu hoàn thành.
