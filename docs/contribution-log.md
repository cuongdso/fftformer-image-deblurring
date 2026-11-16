# Contribution log

Ghi một dòng sau khi task có review/evidence. Không dùng số commit đơn thuần làm thước đo, và không tính riêng report/slides là đóng góp code.

| Member | Workstream | Task ID | PR/commit | Thay đổi kỹ thuật | Test/evidence thực tế | Review status | Tỷ lệ (nếu nhóm thống nhất) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | — | M0 `5b57d61` + M0 foundation follow-up | Nền tảng repository | xem `docs/project-status.md` | pending team review | TBD |
| `DucHai27` | WS1 | M1-WS1-01 | [PR #4](https://github.com/cuongdso/fftformer-image-deblurring/pull/4), merge `7c11ed3`; evidence cập nhật `83b6869` đã tích hợp qua PR #5 | Audit architecture/config/checkpoint và script kiểm chứng | [10/10 CPU checks](research/ws1-01-runtime-evidence.json) | `DONE` — đã merge | Chưa thống nhất |
| `DucHai27` | WS1 | M1-WS1-02 | [PR #5](https://github.com/cuongdso/fftformer-image-deblurring/pull/5), merge `ae64582`; bản bàn giao `83b6869` | ModelSpec, adapter, validation và bảo vệ caller defaults | [30/30 CPU tests tại bản WS1-02](modules/model.md); [CI PR thành công](https://github.com/cuongdso/fftformer-image-deblurring/actions/runs/43812286485) | `DONE` — đã merge | Chưa thống nhất |
| `DucHai27` | WS1 | M1-WS1-03 | [PR #6](https://github.com/cuongdso/fftformer-image-deblurring/pull/6), merge `32d3a2e`; bản bàn giao `b1dcc1a` | Strict loader, incompatibility tests, checkpoint identity và reporter | [55/55 CPU tests](research/ws1-03-runtime-evidence.json); [CI sau merge thành công](https://github.com/cuongdso/fftformer-image-deblurring/actions/runs/43812430543); [handoff](research/ws1-handoff.md) | `DONE` — đã merge | Chưa thống nhất |
