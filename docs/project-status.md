# Trạng thái dự án

## Tổng kết triển khai trên `main`

| Workstream | Trạng thái triển khai | Bằng chứng tích hợp |
| --- | --- | --- |
| WS1 Model | `DONE` — adapter và strict checkpoint loader | PR #4–#6, xem hồ sơ WS1 bên dưới |
| WS2 Data | `DONE` — đọc ảnh, manifest và dataset loader | [PR #8](https://github.com/cuongdso/fftformer-image-deblurring/pull/8), `e482883` |
| WS3 Inference | `DONE` — pipeline và CLI ảnh đơn/thư mục/manifest | [PR #9](https://github.com/cuongdso/fftformer-image-deblurring/pull/9), `c47d58d` |
| WS4 Evaluation | `DONE` — PSNR/SSIM, cấu hình, CLI và kiểm thử tích hợp | [PR #10](https://github.com/cuongdso/fftformer-image-deblurring/pull/10), `dfd1d13` |

Bằng chứng runtime gồm GoPro strict load/forward trên CPU qua API nhóm, GoPro forward trên CUDA qua CLI nhóm, smoke WS2→WS1→WS3 với ảnh tổng hợp và kiểm thử WS2→WS3→WS4 bằng `MockModel`. Tái hiện benchmark đầy đủ GoPro/RealBlur và chạy `test.py` upstream vẫn `UNVERIFIED`. Training và demo UI nằm ngoài phần triển khai đã merge.

## M0 — Repository Foundation

| Hạng mục | Trạng thái | Evidence |
| --- | --- | --- |
| Baseline upstream pin | DONE | submodule `third_party/FFTformer` tại `01306b5…` |
| Audit code/source | DONE (static audit) | `docs/upstream-audit.md` |
| Team/agent onboarding | DONE | `AGENTS.md`, `docs/onboarding.md` |
| Workstream, ownership, contracts | DONE (M0 candidate `PROPOSED`; được M1.0 supersede bởi v1.0 `APPROVED`) | `docs/ownership.md`, `docs/contracts.md`, `docs/decisions/0002-m1-contract-baseline.md` |
| Tutorial reproduction | DONE (audit-derived, chưa chạy) | `docs/reproduction/` |
| Tooling không GPU | DONE | `scripts/check_env.py`, unit test |
| CI lightweight | CONFIGURED (M0); run success sau transfer | `.github/workflows/ci.yml`; Actions run `43360793495` (commit `fd918794`) |
| Baseline inference | NOT RUN | không tải/chạy checkpoint ở M0 |
| Benchmark/reproduction | NOT RUN | không có dataset hoặc metric group |
| Training/fine-tuning/demo | NOT STARTED | ngoài phạm vi M0 |

`DONE` mô tả artifact M0 đã tồn tại, không đồng nghĩa inference/benchmark đã thành công. Chỉ cập nhật bảng này cùng evidence (command, log, PR/commit) khi trạng thái thay đổi.

## M1.0 — Contract alignment

- **Trạng thái:** `APPROVED` — contract baseline v1.0 do Tech Lead chốt; M1.0 documentation finalization complete.
- **Evidence:** [contracts](contracts.md), [M1.0 decision](m1-interface-review.md), [ADR-0002](decisions/0002-m1-contract-baseline.md) và [M1 backlog](backlog.md).
- **Scope:** chốt ranh giới model/data/inference/evaluation và giải vòng dependency. Không có inference/training/benchmark hoặc implementation nào đã chạy.

## Tổng kết WS1 — hoàn thành và đã merge

- **Owner:** `DucHai27`; **kết quả:** 3/3 task `M1-WS1-01/02/03` hoàn thành (`DONE`) và đã tích hợp vào `main` ngày 16/11/2026.
- **Tích hợp:** Owner thực hiện merge theo chỉ đạo leader đã xác nhận; WS1-02 merge trước WS1-03. Trạng thái merged được kiểm tra trực tiếp qua GitHub và lịch sử Git.
- **Kiểm chứng:** WS1-01 đạt 10/10 CPU checks; bộ WS1-02/03 đạt 55/55 tests, với evidence và giới hạn bên dưới. Hoàn thành WS1 không đồng nghĩa đã hoàn tất pipeline toàn dự án hoặc benchmark.

| Task | Hoàn thành công việc | Tích hợp vào `main` |
| --- | --- | --- |
| M1-WS1-01 | `DONE` | [PR #4 đã merge](https://github.com/cuongdso/fftformer-image-deblurring/pull/4), merge commit `7c11ed3`; bản cập nhật evidence `83b6869` đã tích hợp qua PR #5. |
| M1-WS1-02 | `DONE` | [PR #5 đã merge](https://github.com/cuongdso/fftformer-image-deblurring/pull/5), merge commit `ae64582`; bản bàn giao `83b6869`. |
| M1-WS1-03 | `DONE` | [PR #6 đã merge](https://github.com/cuongdso/fftformer-image-deblurring/pull/6), merge commit `32d3a2e`; bản bàn giao `b1dcc1a`. |

Tại lần kiểm tra ngày 16/11/2026, `origin/main` ở `32d3a2e` và chứa cả hai commit bàn giao `83b6869`, `b1dcc1a`; nội dung cây nguồn của `main` khớp bản WS1-03 cuối. [CI PR #5](https://github.com/cuongdso/fftformer-image-deblurring/actions/runs/43812286485), [CI PR #6](https://github.com/cuongdso/fftformer-image-deblurring/actions/runs/43812375759) và [CI sau merge #6](https://github.com/cuongdso/fftformer-image-deblurring/actions/runs/43812430543) đều `completed / success`. Đây là CI nhẹ; kết quả model/checkpoint CPU nằm trong evidence riêng bên dưới.

## M1 — WS1-01 architecture/checkpoint audit

- **Trạng thái:** `DONE` — báo cáo đã merge tại [PR #4](https://github.com/cuongdso/fftformer-image-deblurring/pull/4), commit `237cbf7`; Owner `DucHai27`.
- **Evidence cập nhật:** chạy lại `python -B scripts/ws1_01_verify.py --repo . --output docs/research/ws1-01-runtime-evidence.json` — 10/10 PASS trên CPU. JSON được tái tạo để SHA-256 của script khớp nội dung đã version; dùng LF khi ghi file.
- **Phạm vi:** mapping paper/code, audit constructor/config/checkpoint và shape/compatibility test plan tại [báo cáo](research/fftformer-architecture.md). CPU synthetic smoke không phải benchmark chất lượng.

## M1 — WS1-02 model specification/adapter

- **Trạng thái:** `DONE`, đã merge [PR #5](https://github.com/cuongdso/fftformer-image-deblurring/pull/5) vào `main`, merge commit `ae64582`.
- **Đã có:** `ModelConfig`, `ModelSpec`, `ModelMetadata`, `ModelAdapter`, `build_model` trong `src/fftformer_lab/model/`; khởi tạo upstream tường minh, model `eval()`, validation C-02 và metadata model chưa nạp trọng số.
- **Evidence tại bản WS1-02 sau sửa lỗi dtype/device ngày 16/11/2026:** `python -B -m unittest discover -s tests -p "test_*.py"` chạy 14 test thành công, skip 16; bật `FFTFORMER_RUN_MODEL_TESTS=1` và chạy cùng lệnh thêm `-v`: toàn bộ 30 test thành công trên CPU, gồm 4 test hồi quy default dtype/device của caller. Chi tiết môi trường, scope và test source tại [model guide](modules/model.md#kiểm-thử-ws1-02-trước-khi-bổ-sung-loader).
- **Giới hạn bản WS1-02:** chưa có checkpoint loader, chưa chạy checkpoint qua adapter; phần đó được bổ sung riêng trong WS1-03 bên dưới. Không suy ra baseline reproduction hay pipeline WS2/WS3/WS4 đã hoàn tất.

## M1 — WS1-03 strict checkpoint loader

- **Trạng thái:** `DONE`, đã merge [PR #6](https://github.com/cuongdso/fftformer-image-deblurring/pull/6) vào `main`, merge commit `32d3a2e`.
- **Đã có:** `load_model` với raw tensor state-dict schema validation, key/shape/dtype checks, `weights_only=True`, `map_location` theo device đã resolve và strict load; checkpoint identity ghi trong metadata của adapter ở `eval()`.
- **Evidence sau sửa lỗi review:** `python -B scripts/ws1_03_verify.py --repo . --output docs/research/ws1-03-runtime-evidence.json` — 55/55 test thành công trên CPU, gồm hồi quy dtype/device, GoPro thật dưới caller defaults khác và 8 test reporter/subtest, bao gồm subtest trùng nhãn. [JSON](research/ws1-03-runtime-evidence.json) ghi từng test và metadata/hash thực tế. Chạy `python -B -S -m unittest discover -s tests -p "test_*.py"`: 22 test qua, 33 test opt-in được skip.
- **Artifact:** GoPro checkpoint sẵn có trong upstream submodule, SHA-256 `336717b7c4e97b66af112cc7878106cb185b698bd872c2a8247911507540b2a3`; strict load + forward hai shape qua API nhóm. Fixture khác là Conv2d mock tự tạo, không phải weights FFTformer. Không thêm dataset/checkpoint vào Git nhóm.
- **Phạm vi lần kiểm chứng WS1-03:** CPU với checkpoint GoPro. Lượt CUDA sau đó được ghi trong mục WS3 bên dưới. Checkpoint RealBlur, benchmark chất lượng và training vẫn chưa được xác minh.
- **Bàn giao:** Owner `DucHai27`; [hồ sơ WS1-01/02/03](research/ws1-handoff.md) có files, API, commands/results, PR/merge commit và CI. API WS1 đã có trong `main` để WS3/WS4 sử dụng theo dependency của từng task.

## M1 — WS2 image boundary và dataset loader

- **Owner:** `phanthek2`, nhận WS2 theo phân công đã xác nhận ngày 18/11/2026.
- **Trạng thái:** `M1-WS2-02/03` đã merge vào `main` qua PR #8, merge commit
  `e482883`; báo cáo audit `M1-WS2-01` được bàn giao cùng hồ sơ.
- **Commits code:** WS2-02 `a24c62b`, WS2-03 `0c108a6`; [handoff](research/ws2-handoff.md)
  ghi SHA đầy đủ và evidence kiểm chứng của implementation đã merge.
- **Đã có:** RGB image reader, validation, normalization và explicit BGR conversion;
  deterministic paired/single manifests, JSON relative paths và lazy C-03 `Sample`
  loader trong `src/fftformer_lab/data/`. Không đổi contracts, upstream pin hoặc
  nghiệp vụ WS1/WS3/WS4. Fixtures tự tạo là 3 PNG tổng 224 bytes.
- **Evidence ngày 18/11/2026:** 40/40 WS2 tests qua; toàn bộ 137/137 repository tests
  qua trên CPU khi bật model/checkpoint markers, không skip. Chạy standard-library
  suite: 50 qua, 87 dependency/opt-in tests skip. Compile và whitespace check qua.
- **Boundary smoke thực:** đọc 3 ảnh synthetic tạm qua WS2 → strict GoPro
  checkpoint WS1 → pipeline WS3 trên CPU; ID/shape/range/finiteness và bytes nguồn
  được kiểm tra. Không có đo PSNR/SSIM hoặc benchmark dataset.
- **Ảnh tác giả đã kiểm:** Flower/Istanbul (hai pair RGB 1280x720) từ repo GoPro
  chính thức, tổng 5,016,042 bytes; paired/non-GT outputs và source hashes qua.
  Xem [dataset audit](research/ws2-dataset-audit.md) cho nguồn, terms và layout.
- **Giới hạn:** chưa kiểm toàn bộ GoPro/RealBlur split, CUDA, LMDB, training hoặc
  quality benchmark. Không suy đoán split/CRF của các ảnh ví dụ.
- [Data guide](modules/data.md), [handoff](research/ws2-handoff.md) và
  [runtime evidence](research/ws2-runtime-evidence.json) ghi API, lệnh, phạm vi và
  fingerprint source/test/fixture của bản local.
  Lệnh tái lập: `scripts/ws2_verify.py`, tải bốn PNG chỉ khi bật `--download-examples`.

## M1 — WS3 inference pipeline core và CLI orchestration

- **Owner:** `Alex`.
- **Trạng thái:** `M1-WS3-02` và `M1-WS3-03` đã merge qua [PR #9](https://github.com/cuongdso/fftformer-image-deblurring/pull/9), commit `c47d58d`.
- **Đã có:** `InferencePipeline`, `InferenceRequest`, `InferenceResult`, `PaddingMetadata`, `MockModel`, `write_image` và CLI runner (`cli.py`, `__main__.py`) trong `src/fftformer_lab/inference/`; reflection padding bội 32 ở cạnh phải/đáy, unpad phục hồi spatial extent gốc $H \times W$, clamp $[0, 1]$, xử lý tuần tự batch bảo toàn sample ID, đo thời gian thực thi (giây); CLI hỗ trợ ảnh đơn lẻ (`--input`), thư mục ảnh (`--input-dir`) và dataset manifest của WS2 (`--manifest`), hỗ trợ lưu metadata JSON (`--save-metadata`).
- **Evidence ngày 18/11/2026:** `python -B scripts/ws3_verify.py --repo . --output docs/research/ws3-02-runtime-evidence.json` — 57/57 tests thành công trên CPU trong 0.23s, gồm 10 pure tests (không cần PyTorch), 10 pipeline integration tests, 22 contract enforcement tests và 15 CLI/IO tests. Toàn bộ mã hash file được lưu tại [JSON](research/ws3-02-runtime-evidence.json). Chạy full test suite toàn repository: 152 tests (119 qua trên CPU, 33 opt-in tests skip).
- **Kiểm chứng CUDA ngày 26/11/2026:** Chạy `python -m fftformer_lab.inference --input data/blur.png --output results/restored.png --checkpoint <đường-dẫn-checkpoint-GoPro> --device cuda --save-metadata` trong thư mục tạm với tệp kiến trúc từ submodule pin `01306b5…` và checkpoint có SHA-256 như mục WS1-03. Với PyTorch `2.11.0+cu128`, CUDA `12.8` và NVIDIA T1200 Laptop GPU, CLI xử lý ảnh RGB tổng hợp 33×35, pad lên 64×64, lưu ảnh RGB 33×35 và ghi `device=cuda:0`, `is_mock=false`.
- **Phạm vi:** lượt CUDA trên dùng một ảnh tổng hợp. Benchmark chất lượng trên bộ test GoPro/RealBlur vẫn `UNVERIFIED`.
- **Bàn giao:** [hồ sơ WS3](research/ws3-handoff.md) có đầy đủ API, files, commands và ranh giới cho consumer WS4.
## M1 — WS4-02 Evaluation Engine

- **Trạng thái:** `DONE`, đã merge qua [PR #10](https://github.com/cuongdso/fftformer-image-deblurring/pull/10), commit `dfd1d13`.
- **Đã có:** registry protocol bất biến, evaluator theo sample/batch, PSNR/SSIM CPU variant, provenance C-05, validation GT/ID/shape/dtype/range/finiteness và exclusion rõ ràng cho sample không GT.
- **Protocol:** `upstream-basicsr` cần adapter BasicSR/CUDA tường minh cho `ssim3d=True`; `group-basicsr-cpu` dùng SSIM Gaussian 2D `ssim3d=False` là protocol riêng; `realblur-official` trả unavailable vì chưa có ECC/file workflow.
- **Kiểm chứng:** unit tests WS4-02 kiểm protocol, conversion RGB→BGR/HWC/uint8, PSNR reference, SSIM golden fixture, aggregate và lỗi input; không phải benchmark dataset hay reproduction.
- **Giới hạn:** chưa chạy BasicSR CUDA, RealBlur official evaluator hoặc benchmark GoPro/RealBlur. Integration WS4-04 được kiểm bằng `MockModel`, không phải FFTformer thật.

## M1 — WS4-03 Configuration, Metadata & CLI

- **Trạng thái:** `DONE`, đã merge qua [PR #10](https://github.com/cuongdso/fftformer-image-deblurring/pull/10), commit `dfd1d13`.
- **Đã có:** C-06 JSON schema/parser bất biến, metadata JSON strict và CLI validate/run cho paired prediction/GT images có sẵn.
- **Kiểm chứng:** CLI đã chạy CPU trên hai cặp PNG RGB synthetic với protocol `group-basicsr-cpu`; đây không phải inference FFTformer hay benchmark.
- **Giới hạn:** không có inference FFTformer, benchmark hay RealBlur official evaluator.

## M1 — WS4-04 Integration

- **Trạng thái:** `DONE`, đã merge qua [PR #10](https://github.com/cuongdso/fftformer-image-deblurring/pull/10), commit `dfd1d13`.
- **Đã có:** integration CPU dùng public WS2 manifest/dataset, WS3 pipeline/MockModel và WS4 metric engine trên paired synthetic RGB files; kiểm WS3 padding/crop, batch order, exclusions, errors và provenance mock.
- **Giới hạn:** `MockModel` không phải FFTformer; integration gate dùng fixture CPU. Lượt CUDA với checkpoint GoPro được ghi riêng ở mục WS3. Benchmark và RealBlur official evaluator vẫn `UNVERIFIED`.

## Milestone đề xuất trong kế hoạch ban đầu

- `M1` — Official Baseline Reproduction
- `M2` — Dataset & Evaluation Pipeline
- `M3` — Model Analysis & Inference Integration
- `M4` — Experiments & Demonstration
- `M5` — Final Validation & Academic Deliverables

Danh sách này ghi lại kế hoạch ban đầu. Trạng thái triển khai và bằng chứng hiện hành nằm trong bảng tổng kết WS1–WS4 ở đầu trang.
