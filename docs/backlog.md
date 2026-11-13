# Backlog ban đầu

Các task sau là `PLANNED`, chưa phải issue GitHub đã tạo. Khi tạo issue, dùng template phù hợp và gán owner/reviewer thật.

| ID | WS | Task | Dependency | Acceptance evidence |
| --- | --- | --- | --- | --- |
| M1-WS1-01 | WS1 | Audit chi tiết block/config/checkpoint compatibility với upstream pin | upstream audit | mapping paper↔code + shape/test plan reviewed |
| M1-WS2-01 | WS2 | Xác minh nguồn/license và layout dataset target | dataset owner | manifest/layout test plan, không tải dataset lớn vào Git |
| M1-WS3-01 | WS3 | Thiết kế wrapper inference không làm thay đổi baseline | C-01/C-02 approved | CLI proposal + error/device policy |
| M1-WS4-01 | WS4 | Đối chiếu RealBlur evaluation protocol và experiment schema | dataset/checkpoint source | protocol audit + fixture plan |
| M1-INT-01 | WS4 + tất cả | Chốt C-01..C-06 | bốn task trên | contracts approved, owner/reviewer recorded |

Không bắt đầu M1 chỉ từ backlog này; task cần được nhóm ưu tiên, gán người và review.
