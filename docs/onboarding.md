# Onboarding cho thành viên và AI agent

## Quy trình bắt đầu task

1. Clone với submodule: `git clone --recurse-submodules …`, hoặc chạy `git submodule update --init --recursive`.
2. Đọc `AGENTS.md`, `README.md`, `docs/architecture.md`, `docs/project-status.md` và `docs/ownership.md`.
3. Đọc `docs/modules/<module>.md` và `docs/contracts.md` cho phạm vi task; với upstream đọc thêm audit/reproduction.
4. Xác nhận issue/task ID, owner, reviewer, dependency và out-of-scope. Không sửa khi scope/interface còn mơ hồ.
5. Tạo branch từ `main` mới nhất bằng prefix đúng. Kiểm tra code/test hiện có trước khi sửa.
6. Chạy `python scripts/check_env.py`, sau đó các test liên quan. Không tuyên bố test không chạy là PASS.
7. Cập nhật docs/contract/experiment metadata bị ảnh hưởng. Mở PR theo template và chờ review.

## Prompt giao việc ngắn cho agent

```text
Task ID: <id>; workstream: <WS1..WS4>; branch: <prefix/name>.
Đọc AGENTS.md, docs/project-status.md, docs/ownership.md, docs/contracts.md và docs/modules/<module>.md trước.
Mục tiêu: <đầu ra kỹ thuật có thể kiểm chứng>.
Không làm: <out-of-scope>.
Interface liên quan: <C-xx>; nếu PROPOSED hoặc không rõ, dừng trước khi thay đổi shared API.
Chạy: <commands>. Handoff: files, interface changes, commands/results thực tế, limitations, commit/PR, next steps.
```

## Mẫu handoff

| Mục | Nội dung bắt buộc |
| --- | --- |
| Task/owner/reviewer | ID và workstream |
| Files/modules | Danh sách chính xác |
| Interface | C-xx thay đổi hoặc `Không` |
| Verification | Lệnh đã chạy + kết quả thực tế |
| Hạn chế | `UNVERIFIED`, blocker, rủi ro |
| Traceability | commit SHA/PR, config/checkpoint/dataset refs nếu có |
| Next handoff | dependency hoặc action của người sau |

## Chống context drift

Không copy transcript AI vào repository. `docs/project-status.md`, contracts, ADR và upstream audit là các điểm chuẩn cho từng loại sự thật. Kế hoạch không phải code; không ghi module/metric/dataset là có sẵn nếu không có evidence. Cập nhật đúng tài liệu bị ảnh hưởng trong cùng PR.
