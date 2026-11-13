# Backlog M1 — triển khai theo dependency không vòng lặp

Các task là `PLANNED`, chưa phải issue GitHub đã tạo. Tech Lead đã chốt contract v1.0; Owner/Reviewer ở `docs/ownership.md` vẫn `TBD` và phải được ghi rõ trên từng implementation PR trước khi bắt đầu code.

## Wave A — nghiên cứu/thiết kế song song (evidence cho v1.0)

| ID | Owner role | Reviewer role | Task | Dependency | Acceptance criteria |
| --- | --- | --- | --- | --- | --- |
| M1-WS1-01 | WS1 | WS3 | Audit block/config/checkpoint compatibility với upstream pin, đề xuất C-01 | upstream audit | mapping paper↔code + shape/checkpoint test plan |
| M1-WS2-01 | WS2 | WS1, WS3, WS4 | Xác minh nguồn/license, paired layout và preprocessing, đề xuất C-02/C-03 | upstream audit | manifest/layout + fixtures/validation plan; không commit dataset |
| M1-WS3-01 | WS3 | WS1, WS2, WS4 | Thiết kế wrapper inference/CLI, cung cấp evidence cho C-01/C-02/C-04 | upstream inference audit | CLI/interface proposal + error/device/padding policy, **chưa code shared API** |
| M1-WS4-01 | WS4 | WS1, WS2, WS3 | Audit RealBlur evaluation protocol, experiment schema, đề xuất C-05/C-06 | upstream evaluation audit | protocol audit + metric/config fixture plan |

## Wave B — chốt interface chung

| ID | Owner role | Reviewer role | Task | Dependency | Acceptance criteria |
| --- | --- | --- | --- | --- | --- |
| M1-INT-01 | Tech Lead | Consumer feedback theo C-01..C-06 | Chốt C-01..C-06 v1.0 theo [M1.0 review](m1-interface-review.md) | audit/source evidence Wave A | **DONE:** ADR-0002 + contracts `APPROVED`; không cần đợi gán username |

## Wave C — implementation (chỉ sau contract liên quan APPROVED)

| ID | Owner role | Reviewer role | Task | Dependency | Acceptance criteria |
| --- | --- | --- | --- | --- | --- |
| M1-IMPL-01 | Workstream sở hữu module | Consumer workstream | Bắt đầu code từng module nhỏ dựa trên contract v1.0 | contract liên quan `APPROVED` + scope Tech Lead giao | unit/integration test có bằng chứng, PR theo module |
| M1-BASE-01 | Tech Lead giao | WS1, WS2, WS3, WS4 | Thử official baseline inference bằng checkpoint/dataset/môi trường được xác minh | scope Tech Lead + upstream tutorial + CUDA + artifact hợp lệ | command, output, limitation; không đồng nghĩa benchmark đã tái hiện |

**Sửa vòng phụ thuộc M0:** M1-WS3-01 chỉ làm *design evidence*, nên không còn đòi C-01/C-02 `APPROVED` trước M1-INT-01. M1-INT-01 dùng evidence Wave A để freeze v1.0; gate Wave C hiện do Tech Lead quản lý. Không đánh dấu M1 implementation hoàn tất chỉ vì contract đã được chốt.
