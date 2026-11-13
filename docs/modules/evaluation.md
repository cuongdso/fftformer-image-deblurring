# WS4 — Evaluation, Reproducibility & Integration

**Trạng thái nhận:** `RESERVED` — ưu tiên Tech Lead `cuongdso`; chưa tự-assign owner. **Hiện trạng:** chưa có metric group hay benchmark.

Entry task là `M1-WS4-02`; config/metadata và integration là `M1-WS4-03`/`M1-WS4-04`. Dependency và acceptance criteria chuẩn ở `docs/backlog.md`; hướng dẫn nhận task/handoff ở `INSTRUCTION.md`.

Phạm vi: audit/implement protocol PSNR/SSIM, output/metadata thí nghiệm, light CI và integration gate. Tuân C-05/C-06. Cần đối chiếu evaluator RealBlur official trước khi gọi bất kỳ result nào reproduction.

Test dự kiến: deterministic fixture, input GT mismatch, config validation; tests GPU/checkpoint có marker và artifact quản lý ngoài Git.
