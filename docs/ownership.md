# Ownership theo workstream

GitHub usernames và tên thành viên chưa được cung cấp; các ô Owner/Reviewer để trống có chủ đích. Không có `CODEOWNERS` ở M0 để tránh bịa danh tính hoặc chặn PR sai.

| Workstream | Phạm vi sở hữu | Boundary/interface chính | Owner | Reviewer | Việc đầu tiên đề xuất |
| --- | --- | --- | --- | --- | --- |
| WS1 Model & Architecture | `src/fftformer_lab/model/` | model loading, tensor/model metadata | TBD | TBD | đối chiếu `fftformer_arch.py` với paper và viết shape/checkpoint compatibility plan |
| WS2 Dataset & Preprocessing | `src/fftformer_lab/data/` | paired sample, image normalization/padding | TBD | TBD | đề xuất validator dữ liệu và fixture nhỏ |
| WS3 Inference & Demo | `src/fftformer_lab/inference/`, `app/` | inference request/result, device policy | TBD | TBD | xác định wrapper không phá baseline + CLI contract |
| WS4 Evaluation, Reproducibility & Integration | `src/fftformer_lab/evaluation/`, CI/tooling | metric result, metadata, release gate | TBD | TBD | đối chiếu protocol official và tạo fixture/test plan |

## Quy tắc phối hợp

- Tech Lead là người phê duyệt cuối cùng cho contract chung, dependency liên module và phạm vi task. Mỗi workstream chỉ triển khai task nằm trong scope Tech Lead giao.
- Chủ module chịu trách nhiệm design, docs và test của code trong phạm vi; reviewer xác minh acceptance criteria.
- Người tích hợp có thể đề xuất thay đổi interface, không âm thầm sửa nghiệp vụ workstream khác.
- Task chồng lấn phải có issue nêu owner, dependency và contract trước khi code.
- Đóng góp kỹ thuật đo theo task/evidence đã review, không theo số commit hay slides/README đơn thuần. Log tại `docs/contribution-log.md`.
