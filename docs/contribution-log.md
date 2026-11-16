# Contribution log

Ghi một dòng sau khi task có review/evidence. Không dùng số commit đơn thuần làm thước đo, và không tính riêng report/slides là đóng góp code.

| Member | Workstream | Task ID | PR/commit | Thay đổi kỹ thuật | Test/evidence thực tế | Review status | Tỷ lệ (nếu nhóm thống nhất) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | — | M0 `5b57d61` + M0 foundation follow-up | Nền tảng repository | xem `docs/project-status.md` | pending team review | TBD |
| `DucHai27` | WS1 | M1-WS1-01 | [PR #4](https://github.com/cuongdso/fftformer-image-deblurring/pull/4), `237cbf7`; evidence cập nhật trong branch WS1-02 | Audit architecture/config/checkpoint và script kiểm chứng | [10/10 CPU checks](research/ws1-01-runtime-evidence.json) | PR #4 đã merge; cập nhật evidence chờ review | Chưa thống nhất |
| `DucHai27` | WS1 | M1-WS1-02 | `d24e96a`, `caeebef`; branch `feat/model-model-spec`, PR chưa tạo | ModelSpec, adapter, validation và bảo vệ caller defaults | [30/30 CPU tests tại bản WS1-02](modules/model.md) | Chờ WS3/Tech Lead review | Chưa thống nhất |
| `DucHai27` | WS1 | M1-WS1-03 | `8a83d8f`, `a2b55e1`, `ed77d58`; branch `feat/model-checkpoint-loader`, PR chưa tạo | Strict loader, incompatibility tests, checkpoint identity và reporter | [55/55 CPU tests](research/ws1-03-runtime-evidence.json); [handoff](research/ws1-handoff.md) | Chờ WS3/Tech Lead review | Chưa thống nhất |
