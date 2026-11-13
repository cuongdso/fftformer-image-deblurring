# Backlog M1 — triển khai theo dependency không vòng lặp

Các task là `PLANNED`, chưa phải issue GitHub đã tạo. Owner/Reviewer ở `docs/ownership.md` còn `TBD`, phải gán trước khi review/approve contract.

## Wave A — nghiên cứu/thiết kế song song (không đòi contract APPROVED)

| ID | Owner role | Reviewer role | Task | Dependency | Acceptance criteria |
| --- | --- | --- | --- | --- | --- |
| M1-WS1-01 | WS1 | WS3 | Audit block/config/checkpoint compatibility với upstream pin, đề xuất C-01 | upstream audit | mapping paper↔code + shape/checkpoint test plan |
| M1-WS2-01 | WS2 | WS1, WS3, WS4 | Xác minh nguồn/license, paired layout và preprocessing, đề xuất C-02/C-03 | upstream audit | manifest/layout + fixtures/validation plan; không commit dataset |
| M1-WS3-01 | WS3 | WS1, WS2, WS4 | Thiết kế wrapper inference/CLI theo C-01/C-02/C-04 ở trạng thái `PROPOSED` | upstream inference audit | CLI/interface proposal + error/device/padding policy, **chưa code shared API** |
| M1-WS4-01 | WS4 | WS1, WS2, WS3 | Audit RealBlur evaluation protocol, experiment schema, đề xuất C-05/C-06 | upstream evaluation audit | protocol audit + metric/config fixture plan |

## Wave B — review và chốt interface chung

| ID | Owner role | Reviewer role | Task | Dependency | Acceptance criteria |
| --- | --- | --- | --- | --- | --- |
| M1-INT-01 | WS1–WS4 (facilitator TBD) | Consumer reviewers theo C-01..C-06 | Review chéo và chốt C-01..C-06 theo [M1.0 review](m1-interface-review.md) | proposal/audit Wave A, owner/reviewer đã gán | mỗi contract `APPROVED` qua PR với quyết định, acceptance/test plan, consumer sign-off |

## Wave C — implementation (chỉ sau contract liên quan APPROVED)

| ID | Owner role | Reviewer role | Task | Dependency | Acceptance criteria |
| --- | --- | --- | --- | --- | --- |
| M1-IMPL-01 | Workstream sở hữu module | Consumer workstream | Bắt đầu code từng module nhỏ dựa trên contract đã chốt | từng contract liên quan `APPROVED` | unit/integration test có bằng chứng, PR theo module |
| M1-BASE-01 | TBD (nhóm phân công) | WS1, WS2, WS3, WS4 | Thử official baseline inference bằng checkpoint/dataset/môi trường được xác minh | upstream tutorial + môi trường có CUDA + artifact hợp lệ | command, output, limitation; không đồng nghĩa benchmark đã tái hiện |

**Sửa vòng phụ thuộc M0:** M1-WS3-01 chỉ làm *design proposal*, nên không còn đòi C-01/C-02 `APPROVED` trước M1-INT-01. M1-INT-01 dùng kết quả Wave A để duyệt, rồi mới cho phép code API dùng chung. Không merge code liên module trước gate. Không đánh dấu M1 hoàn tất chỉ vì bảng backlog tồn tại.
