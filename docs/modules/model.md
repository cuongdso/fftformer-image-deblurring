# WS1 — Model & Architecture

**Hiện trạng:** đã triển khai bản `M1-WS1-02` trong working branch, chờ review. Owner/reviewer được quản lý tại [ownership](../ownership.md); tài liệu này không thay đổi phân công workstream.

Dependency và acceptance criteria chuẩn ở [backlog](../backlog.md). `M1-WS1-02` phụ thuộc C-01/C-02 đã `APPROVED`; strict checkpoint loader là `M1-WS1-03`, chưa triển khai. Hướng dẫn nhận task/handoff ở [INSTRUCTION.md](../../INSTRUCTION.md).

## Phạm vi WS1-02

Code nhóm tại `src/fftformer_lab/model/` cung cấp đặc tả, khởi tạo và validation để WS3/WS4 sử dụng theo [C-01/C-02](../contracts.md). [Audit WS1-01](../research/fftformer-architecture.md) là bằng chứng cho constructor mặc định và fingerprint source. Adapter không sao chép kiến trúc upstream, không import BasicSR ngầm và không sửa submodule.

| API | Chức năng |
| --- | --- |
| `ModelConfig` | Cấu hình immutable, mặc định theo GoPro `network_g`; `from_mapping()` nhận mapping đã parse, từ chối field lạ và giá trị không hợp lệ. |
| `ModelSpec` | Định danh architecture, implementation/version, đường dẫn source tường minh, config, device, strictness và checkpoint request. |
| `build_model(spec)` | Kiểm fingerprint source đã audit, resolve device, tạo model ở `eval()` và trả `ModelAdapter`; chỉ khởi tạo trọng số ngẫu nhiên trong WS1-02. |
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

Model trong ví dụ **chưa nạp checkpoint**, metadata ghi `initialization="random"`, `checkpoint_loaded=False`, checkpoint path/hash là `None`. Đầu ra chỉ dùng để kiểm tra API/shape; chưa có ý nghĩa đánh giá chất lượng khử mờ. Nếu truyền `ModelSpec(checkpoint=...)`, `build_model` báo `NotImplementedError` nêu rõ WS1-03 thay vì bỏ qua yêu cầu hoặc trả model giả là đã load. WS1-02 chưa hoàn tất luồng loading C-01.

## Validation và lỗi

- `ModelConfigurationError`: architecture/version chưa hỗ trợ, config sai kiểu/giá trị, field lạ, đường dẫn rỗng, device sai cú pháp hoặc `strict=False`.
- `ModelBackendError`: source thiếu/khác fingerprint, dependency thiếu, CUDA không khả dụng/index sai, lỗi constructor/forward hoặc output sai shape/dtype/device/non-finite. Lỗi forward giữ exception gốc và bổ sung shape/device.
- `ModelInputError`: tensor sai kiểu, dtype, NCHW/3 kênh, batch/chiều rỗng, H/W chưa là bội 32, device không khớp, NaN/Inf hoặc ngoài range C-02. Thứ tự RGB là trách nhiệm provider; tensor ba kênh không tự chứng minh đúng color order.

Model-facing output không bị ép vào `[0,1]`; serialization và clamp policy thuộc boundary WS3. Grad/inference context do caller quyết định. Các yêu cầu CUDA được báo lỗi khi không hợp lệ, không tự rơi về CPU.

## Kiểm thử đã chạy

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

Model tests tạo model từ source upstream đã pin bằng trọng số ngẫu nhiên, không đọc checkpoint hoặc dataset. CUDA unavailable/index tests dùng mock; **CUDA forward vẫn `UNVERIFIED`**. CI mặc định chạy 14 test nhẹ và skip 16 model tests; không cần submodule/PyTorch để chạy nhóm nhẹ.

## Phần còn lại

WS1-03 triển khai strict checkpoint loader/schema/key/shape errors và checkpoint identity theo C-01, dựa trên [kế hoạch WS1-01](../research/fftformer-architecture.md#62-checkpoint-loader-và-device-m1-ws1-03). WS3 tiếp tục sở hữu pad/crop, orchestration, CLI và serialize theo C-02/C-04. Chưa có benchmark, kiểm tra checkpoint qua API nhóm hoặc training; các kết quả CPU ở trên chỉ xác minh phần đã triển khai của WS1-02.
