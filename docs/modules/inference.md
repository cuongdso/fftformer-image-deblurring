# WS3 — Inference & Demonstration

**Trạng thái nhận:** `OPEN`; Owner/reviewer: TBD. **Hiện trạng:** chưa có wrapper/CLI/demo.

Entry task là `M1-WS3-02`; CLI/orchestration phụ thuộc provider interfaces là `M1-WS3-03`. Dependency và acceptance criteria chuẩn ở `docs/backlog.md`; hướng dẫn nhận task/handoff ở `INSTRUCTION.md`.

Phạm vi: load checkpoint, single/batch inference, save output, policy device/VRAM/padding và demo sau milestone được phê duyệt. Tuân C-01, C-02, C-04; các khác biệt với `third_party/FFTformer/test.py` phải được ghi rõ và test.

Không nói wrapper CPU/single-image hỗ trợ cho đến khi implementation đã chạy có evidence; baseline test hiện yêu cầu CUDA và target image.
