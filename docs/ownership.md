# Ownership theo workstream

GitHub usernames và tên thành viên chưa được cung cấp, trừ Tech Lead `cuongdso`. Owner/Reviewer chỉ được cập nhật sau khi thành viên tự nhận task qua GitHub Issue hoặc xác nhận với Tech Lead. Không có `CODEOWNERS` để tránh bịa danh tính hoặc chặn PR sai. Quy trình nhận task nằm ở [INSTRUCTION.md](../INSTRUCTION.md).

| Workstream | Trạng thái nhận | Phạm vi sở hữu | Boundary/interface chính | Owner | Reviewer | Entry task |
| --- | --- | --- | --- | --- | --- | --- |
| WS1 Model & Architecture | `OPEN` | `src/fftformer_lab/model/` | C-01 model loading, C-02 tensor/model metadata | TBD | TBD (WS3 consumer + Tech Lead scope) | `M1-WS1-02` |
| WS2 Dataset & Preprocessing | `OPEN` | `src/fftformer_lab/data/`, fixture/test liên quan | C-02 image I/O, C-03 dataset sample | TBD | TBD (WS1/WS3/WS4 consumer + Tech Lead scope) | `M1-WS2-02` |
| WS3 Inference & Demo | `OPEN` | `src/fftformer_lab/inference/`, `src/fftformer_lab/app/` | C-04 inference result; consumes C-01/C-02 | TBD | TBD (WS1/WS2/WS4 consumer + Tech Lead scope) | `M1-WS3-02` |
| WS4 Evaluation, Reproducibility & Integration | `RESERVED` — ưu tiên Tech Lead `cuongdso` | `src/fftformer_lab/evaluation/`, `configs/`, `experiments/`, integration tests/CI scope | C-05 evaluation, C-06 config; consumes C-02/C-03/C-04 | Chưa tự-assign; reservation cho `cuongdso` | Tech Lead/consumer review theo task | `M1-WS4-02` |

## Quy tắc phối hợp

- Thành viên tự nhận **một** WS đang `OPEN` qua Issue/xác nhận Tech Lead trước khi code; không hai người cùng nhận một WS khi chưa thống nhất. WS4 là ưu tiên của `cuongdso`, không suy diễn đồng thuận của thành viên khác.
- Tech Lead là người phê duyệt cuối cùng cho contract chung, dependency liên module và phạm vi task. Mỗi workstream chỉ triển khai task nằm trong scope Tech Lead giao.
- Chủ module chịu trách nhiệm design, docs và test của code trong phạm vi; reviewer xác minh acceptance criteria.
- Người tích hợp có thể đề xuất thay đổi interface, không âm thầm sửa nghiệp vụ workstream khác.
- Task chồng lấn phải có issue nêu owner, dependency và contract trước khi code.
- Đóng góp kỹ thuật đo theo task/evidence đã review, không theo số commit hay slides/README đơn thuần. Log tại `docs/contribution-log.md`.
