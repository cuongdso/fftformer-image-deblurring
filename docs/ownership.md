# Ownership theo workstream

WS1 do `DucHai27` nhận; WS2 do `phanthek2` nhận theo phân công đã xác nhận; WS3 do `Alex` nhận; Tech Lead là `cuongdso`. Reviewer có thể để `TBD` đến khi ghi rõ trong PR. Không có `CODEOWNERS` để tránh bịa danh tính hoặc chặn PR sai. Quy trình nhận task nằm ở [INSTRUCTION.md](../INSTRUCTION.md).

| Workstream | Trạng thái nhận | Phạm vi sở hữu | Boundary/interface chính | Owner | Reviewer | Entry task |
| --- | --- | --- | --- | --- | --- | --- |
| WS1 Model & Architecture | `DONE` — 3/3 task đã merge | `src/fftformer_lab/model/` | C-01 model loading, C-02 tensor/model metadata | `DucHai27` | Tech Lead `cuongdso`: chỉ đạo merge theo xác nhận Owner; WS3 là consumer | `M1-WS1-01/02/03` hoàn thành |
| WS2 Dataset & Preprocessing | `DONE` — M1-WS2-02/03 đã merge PR #8 | `src/fftformer_lab/data/`, fixture/test liên quan | C-02 image I/O, C-03 dataset sample | `phanthek2` | TBD (WS1/WS3/WS4 consumer + Tech Lead scope) | Audit `M1-WS2-01`, implementation `M1-WS2-02/03` hoàn thành |
| WS3 Inference & Demo | M1-WS3-02 `DONE`, M1-WS3-03 `OPEN` | `src/fftformer_lab/inference/`, `src/fftformer_lab/app/` | C-04 inference result; consumes C-01/C-02 | `Alex` | TBD (WS1/WS2/WS4 consumer + Tech Lead scope) | `M1-WS3-02` hoàn thành, `M1-WS3-03` |
| WS4 Evaluation, Reproducibility & Integration | `M1-WS4-02/03 IMPLEMENTED` — chờ review; WS4-04 chưa triển khai | `src/fftformer_lab/evaluation/`, `configs/`, `experiments/`, integration tests/CI scope | C-05 evaluation, C-06 config; consumes C-02/C-03/C-04 | `cuongdso` ưu tiên nhận; chưa xác nhận assignment cuối | Tech Lead/consumer review theo task | `M1-WS4-02/03` |

WS1 đã hoàn thành nội dung được giao và đã tích hợp vào `main` ngày 16/11/2026 qua PR #4 (WS1-01), #5 (WS1-02), #6 (WS1-03). URL PR, merge commit và CI được ghi tại [project status](project-status.md).

WS3 đã hoàn thành task entry `M1-WS3-02` (pipeline core, reflection padding bội 32 và MockModel test double với 42/42 tests PASS trên CPU). Bằng chứng thực nghiệm tại [ws3-02-runtime-evidence.json](research/ws3-02-runtime-evidence.json) và [hồ sơ bàn giao WS3](research/ws3-handoff.md).

## Quy tắc phối hợp

- Thành viên tự nhận **một** WS đang `OPEN` qua trao đổi trong nhóm, rồi cập nhật Owner tại bảng này trước khi code; không hai người cùng nhận một WS khi chưa thống nhất. WS4 là ưu tiên của `cuongdso`, không suy diễn đồng thuận của thành viên khác.
- Tech Lead là người phê duyệt cuối cùng cho contract chung, dependency liên module và phạm vi task. Mỗi workstream chỉ triển khai task nằm trong scope Tech Lead giao.
- Chủ module chịu trách nhiệm design, docs và test của code trong phạm vi; reviewer xác minh acceptance criteria.
- Người tích hợp có thể đề xuất thay đổi interface, không âm thầm sửa nghiệp vụ workstream khác.
- Task chồng lấn phải được thống nhất owner, dependency và contract trong nhóm trước khi code.
- Đóng góp kỹ thuật đo theo task/evidence đã review, không theo số commit hay slides/README đơn thuần. Log tại `docs/contribution-log.md`.
