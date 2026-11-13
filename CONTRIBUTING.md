# Đóng góp

## Workflow

Luôn tạo branch ngắn hạn từ `main` mới nhất. Dùng một trong các prefix: `feat/model-*`, `feat/data-*`, `feat/inference-*`, `feat/evaluation-*`, `fix/*`, `docs/*`, `chore/*`, `experiment/*`. Không dùng branch dài hạn theo tên người.

Mỗi Pull Request (PR) cần nêu task ID trong `docs/backlog.md`, workstream, phạm vi file, ảnh hưởng API/upstream, lệnh test đã chạy và kết quả thực tế, hạn chế và reviewer. Không tự merge thay reviewer; không force push hoặc rewrite lịch sử `main`.

## Baseline và bản quyền

Không chỉnh sửa `third_party/FFTformer/` trong công việc thường ngày. Submodule là baseline MIT của upstream; mọi tái sử dụng mã phải giữ thông báo copyright/license. Dataset và pretrained weights có điều khoản riêng: kiểm tra nguồn phát hành trước khi tải hoặc phân phối.

## Branch protection (thiết lập thủ công)

M0 không tự cấu hình GitHub branch protection vì không xác minh được quyền quản trị qua môi trường hiện tại. Maintainer nên vào GitHub repository settings và bảo vệ `main`: yêu cầu PR, ít nhất một review theo chính sách nhóm, status check CI sau khi ổn định, chặn force push và không cho phép xóa branch. Ghi lại thay đổi quản trị trong `docs/contribution-log.md`.

## Review và handoff

Reviewer đối chiếu scope với `docs/ownership.md` và contract. Thay đổi shared interface cần thống nhất liên workstream và quyết định của Tech Lead trước khi merge. Xem mẫu handoff tại `docs/onboarding.md`.
