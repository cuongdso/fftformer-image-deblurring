# WS1 — Model & Architecture

**Hiện trạng:** đã triển khai `M1-WS1-02` và bản tiếp nối `M1-WS1-03` trên các feature branch, chờ review. Owner/reviewer được quản lý tại [ownership](../ownership.md); tài liệu này không thay đổi phân công workstream.

Dependency và acceptance criteria chuẩn ở [backlog](../backlog.md). `M1-WS1-02` phụ thuộc C-01/C-02 đã `APPROVED`; strict checkpoint loader `M1-WS1-03` phụ thuộc WS1-02. Hướng dẫn nhận task/handoff ở [INSTRUCTION.md](../../INSTRUCTION.md).

## Phạm vi WS1-02

Code nhóm tại `src/fftformer_lab/model/` cung cấp đặc tả, khởi tạo và validation để WS3/WS4 sử dụng theo [C-01/C-02](../contracts.md). [Audit WS1-01](../research/fftformer-architecture.md) là bằng chứng cho constructor mặc định và fingerprint source. Adapter không sao chép kiến trúc upstream, không import BasicSR ngầm và không sửa submodule.

| API | Chức năng |
| --- | --- |
| `ModelConfig` | Cấu hình immutable, mặc định theo GoPro `network_g`; `from_mapping()` nhận mapping đã parse, từ chối field lạ và giá trị không hợp lệ. |
| `ModelSpec` | Định danh architecture, implementation/version, đường dẫn source tường minh, config, device, strictness và checkpoint request. |
| `build_model(spec)` | Kiểm fingerprint source đã audit, resolve device, tạo model ở `eval()` và trả `ModelAdapter`; chỉ khởi tạo trọng số ngẫu nhiên trong WS1-02. |
| `load_model(spec)` | Khởi tạo model và nạp raw tensor state-dict với `weights_only=True`, `map_location` theo device đã resolve, `strict=True`; trả adapter với checkpoint identity. |
| `ModelAdapter(image)` / `.forward(image)` | Validation tensor và gọi model gốc; giữ nguyên đầu ra, không pad/crop/clamp hoặc chuyển dtype/device. |
| `ModelMetadata.to_dict()` | Metadata JSON-serializable: implementation/version, source hash, config, requested/resolved device, số tham số/state-dict và trạng thái checkpoint. |

Constructor giữ mặc định `dim=48`, `num_blocks=(6,6,12)`, refinement 4, expansion 3 và `bias=False`; không thay đổi ngoại lệ của `Fuse` hay key `reduce_chan_level2`. `num_blocks` được biểu diễn bằng đúng ba mức đã audit; phần tử thứ tư không dùng trong constructor upstream không được nhận trong config nhóm.

Source được nhận qua `ModelSpec.architecture_file`, kiểm SHA-256 theo audit WS1-01 trước khi thực thi. Fingerprint xác định nội dung source; không phụ thuộc vị trí checkout hoặc cài đặt package `basicsr`. Source phải giữ nguyên byte, kể cả kiểu xuống dòng LF, để khớp fingerprint đã audit. Package `fftformer_lab.model` và validation spec import được chỉ với standard library; chỉ bước build/forward mới cần PyTorch và `einops`.

Khởi tạo model dùng context CPU tường minh, sau đó chuyển sang device đã resolve với dtype `torch.float32` theo C-02. Cách này không phụ thuộc default dtype/device của chương trình gọi và không thay đổi các thiết lập đó, kể cả khi constructor thất bại.

## Ví dụ khởi tạo cho kiểm tra shape

Từ repository root, thiết lập đường dẫn package và mở Python:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -B
```

```python
from pathlib import Path
import torch
from fftformer_lab.model import ModelSpec, build_model

spec = ModelSpec(
    architecture_file=Path("third_party/FFTformer/basicsr/models/archs/fftformer_arch.py"),
    device="cpu",
)
adapter = build_model(spec)
print(adapter.metadata.to_dict())
with torch.inference_mode():
    output = adapter(torch.zeros(1, 3, 32, 32, dtype=torch.float32))
```

Model trong ví dụ **chưa nạp checkpoint**, metadata ghi `initialization="random"`, `checkpoint_loaded=False`, checkpoint path/hash là `None`. Đầu ra chỉ dùng để kiểm tra API/shape; chưa có ý nghĩa đánh giá chất lượng khử mờ. Khi có checkpoint request, dùng `load_model` ở phần WS1-03 dưới đây; `build_model` vẫn từ chối checkpoint bằng `NotImplementedError` để tránh bỏ qua yêu cầu nạp trọng số.

## Validation và lỗi

- `ModelConfigurationError`: architecture/version chưa hỗ trợ, config sai kiểu/giá trị, field lạ, đường dẫn rỗng, device sai cú pháp hoặc `strict=False`.
- `ModelBackendError`: source thiếu/khác fingerprint, dependency thiếu, CUDA không khả dụng/index sai, lỗi constructor/forward hoặc output sai shape/dtype/device/non-finite. Lỗi forward giữ exception gốc và bổ sung shape/device.
- `ModelInputError`: tensor sai kiểu, dtype, NCHW/3 kênh, batch/chiều rỗng, H/W chưa là bội 32, device không khớp, NaN/Inf hoặc ngoài range C-02. Thứ tự RGB là trách nhiệm provider; tensor ba kênh không tự chứng minh đúng color order.

Model-facing output không bị ép vào `[0,1]`; serialization và clamp policy thuộc boundary WS3. Grad/inference context do caller quyết định. Các yêu cầu CUDA được báo lỗi khi không hợp lệ, không tự rơi về CPU.

## Kiểm thử WS1-02 trước khi bổ sung loader

Ngày 16/11/2026, môi trường Python 3.14.10, PyTorch 2.15.0+cpu, einops 0.8.2 trên Windows:

```powershell
python -B -m unittest discover -s tests -p "test_*.py"
# Ran 30 tests; OK (skipped=16): 14 chạy thành công, 16 model tests chưa bật.

$env:FFTFORMER_RUN_MODEL_TESTS = '1'
python -B -m unittest discover -s tests -p "test_*.py" -v
# Ran 30 tests; OK: toàn bộ 30 chạy thành công.
Remove-Item Env:FFTFORMER_RUN_MODEL_TESTS
```

- [13 unit tests mới](../../tests/unit/test_model_spec.py): constructor/config validation, immutable config/metadata, identity/device/strictness, checkpoint request, source thiếu/thay đổi, import không dependency.
- [16 model tests opt-in](../../tests/model/test_model_adapter.py): constructor metadata/eval, shape `32x32`, `64x96`, batch 2, so sánh đầu ra với model upstream, input không bị sửa, các đầu vào sai và output lỗi. Bốn test hồi quy kiểm tra caller mặc định float64, device meta, kết hợp cả hai và phục hồi defaults khi constructor thất bại. Kiểm tra không clamp và truyền lỗi dùng mock có tên rõ trong test.
- 1 test tooling có sẵn trong `tests/unit/test_check_env.py`.

Model tests WS1-02 tạo model từ source upstream đã pin bằng trọng số ngẫu nhiên, không đọc checkpoint hoặc dataset. CUDA unavailable/index tests dùng mock; **CUDA forward vẫn `UNVERIFIED`**. Tại bản WS1-02 sau sửa lỗi defaults, CI mặc định chạy 14 test nhẹ và skip 16 model tests; không cần submodule/PyTorch để chạy nhóm nhẹ.

## WS1-03 — Strict checkpoint loader

`load_model` nhận `ModelSpec` có checkpoint path tường minh, chỉ hỗ trợ schema đã audit: mapping không rỗng với key chuỗi và giá trị tensor. Wrapper như `params`, `params_ema`, `state_dict` và prefix key khác không được tự đoán/chuyển đổi. Trước khi nạp, loader đối chiếu toàn bộ key, shape và dtype để tránh cast âm thầm; sau đó vẫn gọi `load_state_dict(strict=True)`. Khi mismatch, thông báo có checkpoint/model/device, tổng số lỗi và tối đa 10 key/shape/dtype làm ví dụ.

| Lỗi | Trường hợp |
| --- | --- |
| `CheckpointError` | Không đọc được đường dẫn, file là thư mục, file hỏng hoặc không deserialize được bằng `weights_only=True`. |
| `CheckpointSchemaError` | Không phải raw tensor state-dict được hỗ trợ. |
| `CheckpointCompatibilityError` | Thiếu/thừa key, sai shape/dtype hoặc strict load thất bại. |

Model được tạo mới trước khi load; model đang dùng của caller không bị nạp dở. Checkpoint chỉ được mở đọc, hash và deserialize trên cùng file handle. Metadata thành công có `checkpoint_loaded=True`, `initialization="checkpoint"`, checkpoint path/SHA-256/size/schema và `checkpoint_map_location`. Không chuyển `strict=False` hoặc fallback device khi có lỗi.

Sau khi thiết lập `PYTHONPATH=src` như ví dụ trên:

```python
from pathlib import Path
from fftformer_lab.model import ModelSpec, load_model

adapter = load_model(ModelSpec(
    architecture_file=Path("third_party/FFTformer/basicsr/models/archs/fftformer_arch.py"),
    checkpoint=Path("third_party/FFTformer/pretrain_model/fftformer_GoPro.pth"),
    device="cpu",
))
print(adapter.metadata.to_dict())
```

## Kiểm thử WS1-03 và evidence hiện tại

[Script kiểm chứng](../../scripts/ws1_03_verify.py) chạy cả bộ test WS1-02/03 và xuất [JSON evidence](../research/ws1-03-runtime-evidence.json), gồm kết quả từng test, môi trường, metadata checkpoint và SHA-256 các file code/test/script:

```powershell
python -B scripts/ws1_03_verify.py --repo . --output docs/research/ws1-03-runtime-evidence.json
# Summary: 53/53 passed
```

Kết quả thực tế trên Windows, Python 3.14.10, PyTorch 2.15.0+cpu, einops 0.8.2, 2 CPU threads:

- 30 test của WS1-02/tooling tiếp tục qua, gồm 4 test hồi quy bảo vệ default dtype/device của caller.
- [14 test fixture mới](../../tests/model/test_checkpoint_loader.py) dùng model Conv2d mock và checkpoint cực nhỏ do test tự tạo trong thư mục tạm; không phải checkpoint FFTformer.
- 3 integration tests dùng checkpoint GoPro thật sẵn có: strict load và forward `32x32`/`64x96` hữu hạn; cấu hình lệch `dim=32` bị từ chối với lỗi tensor shape; caller mặc định `float64` + `meta` vẫn load/forward được trên CPU float32 và giữ nguyên defaults.
- [6 test reporter không cần dependency](../../tests/unit/test_ws1_03_verify.py): ghi nhận parent test và subtest lỗi/skip, nhiều lỗi trong một parent, test thường failure/error và suite rỗng. Các failure/error bên trong là tình huống được tạo có chủ đích để kiểm chứng reporter; cả 6 test bên ngoài đã thành công.
- Chạy không dependency ngoài standard library: `python -B -S -m unittest discover -s tests -p "test_*.py"` — 20 test qua, 33 test model/checkpoint được skip.

Các model/fixture tests được bật bởi `FFTFORMER_RUN_MODEL_TESTS=1`; ba test GoPro còn yêu cầu `FFTFORMER_GOPRO_CHECKPOINT` trỏ đến artifact. Script đặt cả hai marker tường minh; CI mặc định không chạy model/checkpoint tests. Có thể dùng `--checkpoint` trỏ tới bản sao artifact ở ngoài repository và `--output` khác để giữ lần evidence cũ. JSON sử dụng đường dẫn tương đối với repository, thư mục external được ẩn và xuống dòng LF.

Reporter giữ một mục `checks` cho mỗi parent test; chi tiết subtest lỗi/skip nằm riêng tại `subtests`. `summary.total/passed` đếm parent tests, còn `failed/errors/skipped` là số sự kiện tương ứng của unittest, có thể nhiều hơn số parent khi một test có nhiều subtest lỗi. Suite rỗng, thiếu bản ghi, có lỗi hoặc bị skip đều không được báo audit `PASS`.

Chỉ kết luận tương thích cho GoPro checkpoint SHA-256 `336717b7c4e97b66af112cc7878106cb185b698bd872c2a8247911507540b2a3` với source/config/device đã ghi trong JSON. Không có checkpoint mới được tải hoặc thêm vào Git của nhóm.

## Phần còn lại

WS3 tiếp tục sở hữu pad/crop, orchestration, CLI và serialize theo C-02/C-04. **CUDA forward, checkpoint RealBlur, benchmark chất lượng và training vẫn `UNVERIFIED`/chưa chạy.** Các kết quả trên chưa phải baseline reproduction. Hai task WS1-02/03 chờ review qua PR và chưa đánh dấu `DONE`; WS3 và Tech Lead review consumer impact/scope theo backlog.
