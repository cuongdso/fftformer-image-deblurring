# Bàn giao WS1 — Model & Architecture

Ngày kiểm chứng: 16/11/2026. Owner: `DucHai27`. Phạm vi: `M1-WS1-01`, `M1-WS1-02`, `M1-WS1-03` theo [backlog](../backlog.md).

Reviewer đề xuất: WS3 cho consumer impact (username cần chốt trong PR), Tech Lead `cuongdso` cho scope/interface. Đây là hồ sơ kỹ thuật để review; WS1-02/03 chưa được đánh dấu `DONE` hoặc được reviewer phê duyệt. Phân công tại [ownership](../ownership.md), trạng thái chuẩn tại [project status](../project-status.md).

## Kết quả theo task

| Task | Yêu cầu và kết quả | Evidence |
| --- | --- | --- |
| M1-WS1-01 | Mapping paper/code; audit FSAS/DFFN, constructor/config/checkpoint; kế hoạch shape và compatibility tests. Đã strict-load GoPro và chạy synthetic CPU smoke. | [Báo cáo kiến trúc](fftformer-architecture.md), [10/10 checks](ws1-01-runtime-evidence.json) |
| M1-WS1-02 | ModelSpec/ModelConfig, metadata, adapter và validation C-02; model eval; lỗi source/config/device/input/output có ngữ cảnh; bảo vệ default dtype/device của caller. | [API và tests](../modules/model.md); 30 tests WS1-02/tooling nằm trong bộ 55 tests hiện tại |
| M1-WS1-03 | Loader raw tensor state-dict, kiểm schema/key/shape/dtype, `weights_only=True`, `map_location` theo device và `strict=True`; identity checkpoint rõ. | [55/55 tests](ws1-03-runtime-evidence.json): 30 tests trước đó + 14 fixture tests + 3 GoPro integration tests + 8 reporter tests |

Hai lỗi evidence sau review đã xử lý: WS1-01 tái tạo JSON với hash của script hiện tại, ghi LF; reporter WS1-03 giữ mọi thông báo lỗi/lý do skip khi subtest trùng nhãn. Hai regression tests mới từng thất bại trên reporter cũ và đều thành công sau sửa.

## Files/modules bàn giao

| Nhóm | Files |
| --- | --- |
| WS1-01 audit | `docs/research/fftformer-architecture.md`, `scripts/ws1_01_verify.py`, `docs/research/ws1-01-runtime-evidence.json` |
| WS1-02 model API | `src/fftformer_lab/model/spec.py`, `adapter.py`, `__init__.py`, `README.md` trong cùng module |
| WS1-02 tests | `tests/unit/test_model_spec.py`, `tests/model/test_model_adapter.py`, `tests/model/__init__.py` |
| WS1-03 loader/evidence | `src/fftformer_lab/model/checkpoint.py`, bổ sung exports/metadata trong `__init__.py`/`adapter.py`, `tests/model/test_checkpoint_loader.py`, `scripts/ws1_03_verify.py`, `tests/unit/test_ws1_03_verify.py`, `docs/research/ws1-03-runtime-evidence.json` |
| Hướng dẫn sử dụng/kiểm chứng | `README.md`, `docs/modules/model.md`, `docs/project-status.md`, `docs/reproduction/pretrained-models.md`, `scripts/README.md`, `tests/README.md` |
| Phân công/bàn giao | `INSTRUCTION.md`, `docs/ownership.md`, `docs/backlog.md`, `docs/contribution-log.md`, `docs/research/ws1-handoff.md` |

Thay đổi shared contract: **Không**. Triển khai public Python API theo [C-01/C-02](../contracts.md): `ModelConfig`, `ModelSpec`, `ModelMetadata`, `ModelAdapter`, `build_model`, `load_model` và các exception được export. Model guide có ví dụ gọi và policy lỗi. Không đổi contracts v1.0, upstream source/pin, nghiệp vụ WS2/WS3/WS4 hoặc thêm artifact lớn vào Git.

WS3/WS4 nhận adapter có model ở eval mode và metadata implementation/config/device/checkpoint. `build_model` tạo trọng số ngẫu nhiên dùng cho shape tests; để dùng GoPro weights phải gọi `load_model` với checkpoint path tường minh. Input model-facing là RGB float32 NCHW với H/W là bội 32. WS3 sở hữu pad/crop, inference context, orchestration và serialize/clamp policy.

## Commands và kết quả thực tế

Tại repository root trên bản chứa code commit `ed77d58`:

```powershell
python -B scripts/ws1_01_verify.py --repo . --output docs/research/ws1-01-runtime-evidence.json
# 10/10 PASS; evidence được tạo tại commit WS1-02 83b6869 và giữ nguyên trong WS1-03.

python -B scripts/ws1_03_verify.py --repo . --output docs/research/ws1-03-runtime-evidence.json
# 55/55 PASS; không skip.

python -B -S -m unittest discover -s tests -p "test_*.py"
# 22 PASS, 33 opt-in tests được skip.

python -m compileall -q scripts src tests
# Exit 0.

git diff --check origin/main HEAD
# Không có lỗi whitespace.
```

WS1-01 được chạy khi cập nhật nhánh WS1-02; toàn bộ bộ test WS1-02/03 được chạy sau khi đồng bộ hai nhánh. SHA-256 trong hai JSON được đối chiếu với bytes chuẩn bị commit và bytes đã commit. Bộ model/fixture tests cần `FFTFORMER_RUN_MODEL_TESTS=1`; ba test GoPro còn cần `FFTFORMER_GOPRO_CHECKPOINT`. Verifier đặt hai marker này; CI mặc định chỉ chạy tests nhẹ.

Môi trường: Windows, Python 3.14.10, PyTorch 2.15.0+cpu, einops 0.8.2, PyYAML 6.0.3 cho WS1-01; các verifier dùng 2 CPU threads. Kiểm thử GitHub CI trên Python 3.10/3.11 vẫn chờ mở PR; kết quả local không thay thế trạng thái CI.

Artifact duy nhất dùng cho integration thực: `third_party/FFTformer/pretrain_model/fftformer_GoPro.pth`, có sẵn trong submodule upstream. SHA-256 `336717b7c4e97b66af112cc7878106cb185b698bd872c2a8247911507540b2a3`, 66.441.171 byte, 535 tensors. Upstream pin `01306b5d4072146a011c8c6e9d4927cf4607f1d5`; config `ModelConfig()` theo GoPro `network_g`. Fixture Conv2d được test tự tạo trong thư mục tạm và nhận diện rõ là mock. Không dùng dataset để đo chất lượng.

## Commit, PR và bước tiếp theo

| Task | Commit truy vết | PR |
| --- | --- | --- |
| WS1-01 | Báo cáo `237cbf7`; cập nhật evidence/ownership `83b6869` trong nhánh WS1-02 | [PR #4 đã merge](https://github.com/cuongdso/fftformer-image-deblurring/pull/4); phần cập nhật đi cùng PR WS1-02 |
| WS1-02 | Implementation `d24e96a`, sửa caller defaults `caeebef`, bản bàn giao nhánh `83b6869` | Chưa tạo; [compare main với WS1-02](https://github.com/cuongdso/fftformer-image-deblurring/compare/main...feat/model-model-spec) |
| WS1-03 | Loader `8a83d8f`, sửa evidence/tests `a2b55e1`, sửa reporter và evidence cuối `ed77d58` | Chưa tạo; [xem riêng thay đổi WS1-03 so với WS1-02](https://github.com/cuongdso/fftformer-image-deblurring/compare/feat/model-model-spec...feat/model-checkpoint-loader) |

1. Mở PR WS1-02 vào `main` theo template, ghi Owner/Reviewer, task/contracts, commands/results và limitations từ hồ sơ này. Tiêu đề đề xuất: `feat(model): M1-WS1-02 model specification and adapter`.
2. WS3/Tech Lead review và CI qua trước khi merge theo [CONTRIBUTING](../../CONTRIBUTING.md). WS1-03 đã chứa nhánh WS1-02 mới nhất; người tích hợp kiểm tra base `main` sau khi WS1-02 được merge.
3. Mở PR WS1-03 vào `main`, tiêu đề đề xuất: `feat(model): M1-WS1-03 strict checkpoint loader and evidence`. Link compare giữa hai feature branch chỉ dùng để review phần chênh lệch trước khi WS1-02 merge.
4. Bổ sung URL PR thực, reviewer và kết quả CI/review vào hồ sơ/contribution log khi có. Chỉ cập nhật task `DONE` sau nghiệm thu. WS3 dùng API này cho inference; WS4 tích hợp sau khi các provider liên quan ổn định.

## Hạn chế đã công bố

- `UNVERIFIED`: CUDA forward, checkpoint RealBlur, benchmark PSNR/SSIM, full baseline reproduction, training và pipeline liên workstream. Các phần này không thuộc acceptance criteria còn thiếu của ba task WS1 hiện tại.
- Compatibility chỉ được xác nhận cho GoPro artifact/config/device nêu trên. Forward dùng tensor synthetic; không suy ra chất lượng khử mờ trên dataset.
- Source architecture phải giữ nguyên bytes LF để khớp fingerprint đã audit. Đường dẫn source/checkpoint được consumer cung cấp tường minh.
- Reviewer WS3 và quyết định nghiệm thu của Tech Lead chưa được xác nhận; test tự động và review code không thay thế phê duyệt của nhóm.
