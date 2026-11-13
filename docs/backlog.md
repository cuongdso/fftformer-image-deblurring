# Backlog M1 — triển khai theo dependency không vòng lặp

Các task implementation Wave C là `PLANNED` cho đến khi được nhận trên GitHub Issue hoặc Tech Lead xác nhận. Bốn Issue tự nhận không được tạo bởi branch này: nếu chưa có Issue canonical, `OPEN` không cho phép bắt đầu code; thành viên tạo Issue theo template nếu có quyền hoặc chờ xác nhận có thể dẫn chiếu của Tech Lead. Tech Lead đã chốt contract v1.0; trạng thái tự nhận và Owner/Reviewer nằm ở `docs/ownership.md` và phải được lặp lại trong từng implementation PR. Hướng dẫn đầy đủ cho người/agent nằm ở `INSTRUCTION.md`.

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

## Wave C — implementation tasks (chỉ sau contract liên quan `APPROVED`)

Các task dưới đây không là bằng chứng implementation đã tồn tại. Mỗi task có một PR nhỏ; Owner là người tự nhận hoặc được Tech Lead xác nhận. Mock/fixture chỉ kiểm chứng contract boundary, không chứng minh FFTformer inference.

| ID | Owner role / trạng thái | Reviewer role | Task và file boundary | Dependency không vòng | Acceptance criteria / DoD |
| --- | --- | --- | --- | --- | --- |
| M1-WS1-02 | WS1 / `OPEN` | WS3 consumer + Tech Lead scope | ModelSpec/model adapter tối thiểu trong `src/fftformer_lab/model/` | C-01, C-02 `APPROVED` | metadata implementation/config/device rõ; validation error có ngữ cảnh; unit test shape/config; không sửa upstream |
| M1-WS1-03 | WS1 / sau M1-WS1-02 | WS3 consumer + Tech Lead | Strict checkpoint loader và incompatibility tests trong WS1 | M1-WS1-02; fixture/artifact được phép | thiếu file/schema/key/shape mismatch không bị che; strict-load test; checkpoint compatibility chỉ claim phần đã chạy |
| M1-WS2-02 | WS2 / `OPEN` | WS1, WS3 consumer + Tech Lead | Image validation/normalization trong `src/fftformer_lab/data/` | C-02 `APPROVED` | RGB `float32` CHW `[3,H,W]` `[0,1]`; BGR conversion explicit; range/channel/corrupt-image tests |
| M1-WS2-03 | WS2 / sau M1-WS2-02 | WS4 consumer + Tech Lead | Deterministic paired manifest/loader và non-GT `Sample` | M1-WS2-02; C-03 `APPROVED` | pair/missing/duplicate/size error tests + legal tiny fixtures; `target=None` khi không GT; không commit dataset lớn |
| M1-WS3-02 | WS3 / `OPEN` | WS1, WS2, WS4 consumer + Tech Lead | Inference request/result, pad/crop, mock pipeline tests trong `src/fftformer_lab/inference/` | C-01, C-02, C-04 `APPROVED` | `InferenceResult` một sample, metadata/crop/error tests; mock được nhận diện rõ; không claim real inference |
| M1-WS3-03 | WS3 / sau provider APIs ổn định | WS1, WS2, WS4 + Tech Lead | CLI/orchestration trong inference/app boundary | M1-WS3-02 + public adapters M1-WS1-02, M1-WS2-03 | CLI help/error test, single/batch ordering, handoff provider compatibility; không tự claim CPU/GPU support |
| M1-WS4-02 | WS4 / `RESERVED` ưu tiên `cuongdso` | WS2, WS3 consumer + Tech Lead | Metric/protocol adapter và deterministic fixtures trong `src/fftformer_lab/evaluation/` | C-02, C-05, C-06 `APPROVED` | GT/alignment/protocol/non-finite errors; fixture reference evidence; tách BasicSR/RealBlur official |
| M1-WS4-03 | WS4 / sau M1-WS4-02 | WS1, WS2, WS3 + Tech Lead | Evaluation config, experiment metadata và CLI | M1-WS4-02 | config validation không GPU; metadata contract/config/dataset/checkpoint identity; không claim benchmark |
| M1-WS4-04 | WS4 / sau providers | WS1, WS2, WS3 + Tech Lead | Lightweight integration tests giữa public adapters | M1-WS1-02 + M1-WS2-03 + M1-WS3-03 + M1-WS4-03 | test chỉ các path đã có fixture/mock; marker cho GPU/checkpoint; không đổi interface lặng lẽ |
| M1-BASE-01 | Tech Lead giao riêng | WS1, WS2, WS3, WS4 | Thử official baseline inference với checkpoint/dataset/môi trường hợp lệ | scope Tech Lead + upstream tutorial + CUDA + artifact hợp lệ | command, output, limitation; không đồng nghĩa benchmark đã tái hiện |

**Dependency graph:** WS1-02, WS2-02, WS3-02 và WS4-02 có thể bắt đầu độc lập ngay vì chỉ phụ thuộc contracts đã `APPROVED`. WS1-03 phụ thuộc WS1-02; WS2-03 phụ thuộc WS2-02; WS3-03 đợi public provider adapters; WS4-03 đợi metric adapter; WS4-04 là integration gate cuối. Không có cạnh nào quay ngược từ implementation về approval C-01..C-06. Không đánh dấu M1 implementation hoàn tất chỉ vì contract đã được chốt.
