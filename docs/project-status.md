# Trạng thái dự án

## M0 — Repository Foundation

| Hạng mục | Trạng thái | Evidence |
| --- | --- | --- |
| Baseline upstream pin | DONE | submodule `third_party/FFTformer` tại `01306b5…` |
| Audit code/source | DONE (static audit) | `docs/upstream-audit.md` |
| Team/agent onboarding | DONE | `AGENTS.md`, `docs/onboarding.md` |
| Workstream, ownership, contracts | DONE (thiết kế `PROPOSED`) | `docs/ownership.md`, `docs/contracts.md` |
| Tutorial reproduction | DONE (audit-derived, chưa chạy) | `docs/reproduction/` |
| Tooling không GPU | DONE | `scripts/check_env.py`, unit test |
| CI lightweight | CONFIGURED, chưa quan sát run | `.github/workflows/ci.yml` |
| Baseline inference | NOT RUN | không tải/chạy checkpoint ở M0 |
| Benchmark/reproduction | NOT RUN | không có dataset hoặc metric group |
| Training/fine-tuning/demo | NOT STARTED | ngoài phạm vi M0 |

`DONE` mô tả artifact M0 đã tồn tại, không đồng nghĩa inference/benchmark đã thành công. Chỉ cập nhật bảng này cùng evidence (command, log, PR/commit) khi trạng thái thay đổi.

## Milestone đề xuất

- `M1` — Official Baseline Reproduction
- `M2` — Dataset & Evaluation Pipeline
- `M3` — Model Analysis & Inference Integration
- `M4` — Experiments & Demonstration
- `M5` — Final Validation & Academic Deliverables

Các milestone trên là `PLANNED`, chưa được đánh dấu hoàn thành.
