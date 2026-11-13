# WS2 — Dataset & Preprocessing

**Trạng thái nhận:** `OPEN`; Owner/reviewer: TBD. **Hiện trạng:** chưa có loader nhóm.

Entry task là `M1-WS2-02`; paired manifest/loader là `M1-WS2-03`. Dependency và acceptance criteria chuẩn ở `docs/backlog.md`; hướng dẫn nhận task/handoff ở `INSTRUCTION.md`.

Phạm vi: manifest/layout paired, validation image, crop/padding/normalization/color policy và adapter cho ảnh không GT. Input/output phải theo C-02/C-03. Không commit dataset và không tự coi `BGR` hoặc range là chuẩn nếu chưa duyệt contract.

Test dự kiến: pair match, extension/channel/size invalid, non-destructive preprocess trên fixture nhỏ hợp pháp. Xem `docs/reproduction/datasets.md` cho layout upstream test.
