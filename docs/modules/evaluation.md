# WS4 — Evaluation, Reproducibility & Integration

**Owner/reviewer:** TBD. **M0:** không có metric group hay benchmark.

Phạm vi: audit/implement protocol PSNR/SSIM, output/metadata thí nghiệm, light CI và integration gate. Tuân C-05/C-06. Cần đối chiếu evaluator RealBlur official trước khi gọi bất kỳ result nào reproduction.

Test dự kiến: deterministic fixture, input GT mismatch, config validation; tests GPU/checkpoint có marker và artifact quản lý ngoài Git.
