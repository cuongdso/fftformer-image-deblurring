# WS3 — Inference & Demonstration

**Hiện trạng:** WS3 đã hoàn thành task entry `M1-WS3-02` (Inference pipeline core) đạt trạng thái `DONE`; task `M1-WS3-03` (CLI & Orchestration) ở trạng thái `PLANNED`. Evidence tích hợp và kết quả kiểm thử tại [project status](../project-status.md), phân công tại [ownership](../ownership.md).

Dependency và acceptance criteria chuẩn ở [backlog](../backlog.md). `M1-WS3-02` phụ thuộc vào các contract C-01, C-02, C-04 đã `APPROVED`. Hướng dẫn nhận task/handoff ở [INSTRUCTION.md](../../INSTRUCTION.md).

[Hồ sơ bàn giao WS3](../research/ws3-handoff.md) tổng hợp phạm vi task, files bàn giao, kết quả kiểm chứng 42/42 checks và ranh giới với WS4.

## Phạm vi WS3-02

Mã nguồn tại `src/fftformer_lab/inference/` cung cấp pipeline điều phối suy luận khử mờ, tính toán padding reflection tới bội 32 và khôi phục kích thước gốc theo [Contract C-02/C-04](../contracts.md). Pipeline không đọc ghi file trực tiếp từ đĩa (phần I/O đó thuộc C-02/WS2 và CLI M1-WS3-03). Pipeline nhận `ModelAdapter` từ WS1 (đã ở `eval()`), `MockModel` phục vụ kiểm thử, hoặc một callable/`nn.Module` tương thích.

| API | Chức năng |
| --- | --- |
| `InferencePipeline` | Bộ điều phối suy luận chính; nhận model, xác thực device, hỗ trợ `run_single` và `run_batch` tuần tự, đo thời gian thực thi chính xác. |
| `InferenceRequest` | Dataclass đóng gói tensor ảnh đầu vào `image` (RGB float32 CHW) và `sample_id`. |
| `InferenceResult` | Dataclass immutable chứa ảnh đã khôi phục `restored_image`, `sample_id`, `padding`, các shape trước/sau pad, `device`, `elapsed_time_seconds`, `is_mock` và `to_dict()`. |
| `PaddingMetadata` | Lưu trữ siêu dữ liệu padding: `top`, `bottom`, `left`, `right`, `mode="reflect"`, `original_shape` và `padded_shape`. |
| `calculate_reflection_padding` | Tính toán lượng pad đáy và phải để $H, W$ chia hết cho 32; phát hiện sớm và chặn lỗi nếu kích thước ảnh quá nhỏ cho reflection pad ($H \le \text{pad\_h}$ hoặc $W \le \text{pad\_w}$). |
| `pad_image` | Áp dụng reflection padding lên tensor 3D CHW hoặc 4D NCHW. |
| `unpad_image` | Cắt (crop) tensor về lại spatial extent ban đầu $H \times W$. |
| `validate_input_tensor` | Kiểm tra tensor đầu vào tuân thủ nghiêm ngặt C-02: 3D, RGB 3 kênh, float32, giá trị hữu hạn không NaN/Inf, range $[0.0, 1.0]$. |
| `MockModel` | Test double độc lập hỗ trợ các mode `identity`, `invert`, `shift`; gán nhãn `is_mock = True` và kiểm tra đầu vào NCHW chia hết cho 32. |

## Cơ chế Padding & Cropping (C-02/C-04)

Theo baseline official và Contract C-02, mô hình FFTformer yêu cầu spatial dimensions $H$ và $W$ phải là bội số của 32 (do cấu trúc chia patch 8x8 ở cấp sâu nhất $H/4$).
Pipeline thực hiện padding ở góc phải và cạnh đáy bằng phép phản xạ (*reflection padding*):

$$\Delta H = (32 - H \pmod{32}) \pmod{32}, \quad \Delta W = (32 - W \pmod{32}) \pmod{32}$$

- Bộ pad tương ứng: `(left=0, right=ΔW, top=0, bottom=ΔH)`.
- Khi kích thước đã chia hết cho 32, padding bằng 0 và ảnh được giữ nguyên.
- Sau khi model forward, tensor output có kích thước $[1, 3, H + \Delta H, W + \Delta W]$ được squeeze bỏ chiều batch, sau đó crop lại `[:H, :W]` và clamp trong khoảng $[0.0, 1.0]$.

## Ví dụ sử dụng InferencePipeline

Từ repository root, thiết lập đường dẫn và chạy Python:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -B
```

```python
import torch
from fftformer_lab.inference import InferencePipeline, MockModel, InferenceRequest

# 1. Khởi tạo pipeline với MockModel (hoặc ModelAdapter từ WS1)
mock_model = MockModel(mode="identity", device="cpu")
pipeline = InferencePipeline(mock_model)

# 2. Tạo tensor ảnh hợp lệ C-02 (RGB float32 CHW trong [0, 1])
image = torch.rand(3, 50, 70, dtype=torch.float32)

# 3. Thực thi suy luận cho một ảnh
result = pipeline.run_single(image, sample_id="demo_sample")
print("Input shape:   ", result.input_shape)   # (3, 50, 70)
print("Padded shape:  ", result.padded_shape)  # (3, 64, 96) - bội số 32
print("Output shape:  ", result.output_shape)  # (3, 50, 70) - khôi phục HxW
print("Elapsed time:  ", result.elapsed_time_seconds, "s")
print("Is mock model: ", result.is_mock)       # True

# 4. Thực thi hàng loạt (batch) bảo toàn thứ tự tuần tự
batch_results = pipeline.run_batch([image, image])
print(f"Processed {len(batch_results)} samples in sequence.")
```

## Validation và hệ thống ngoại lệ

- `InferenceError`: Ngoại lệ cơ sở cho toàn bộ module inference.
- `InferenceInputError`: Bắt lỗi khi tensor không phải float32, sai số kênh khác 3, không phải 3D, chứa NaN/Inf, ngoài khoảng $[0, 1]$, hoặc thiết bị yêu cầu mâu thuẫn với thiết bị cố định của model.
- `InferencePaddingError`: Kích thước ảnh quá nhỏ không đủ điều kiện cho reflection padding, tham số âm hoặc unpad vượt quá kích thước tensor.
- `InferenceExecutionError`: Model đang ở chế độ huấn luyện (`model.training == True`), model forward văng lỗi, không chuyển được thiết bị, hoặc output trả về không phải tensor, sai shape, sai dtype hoặc chứa giá trị không hữu hạn.

## Kiểm thử WS3-02 và evidence hiện tại

Script kiểm chứng `scripts/ws3_verify.py` tự động chạy toàn bộ 42 tests của WS3 và xuất [JSON evidence](../research/ws3-02-runtime-evidence.json):

```powershell
python -B scripts/ws3_verify.py --repo . --output docs/research/ws3-02-runtime-evidence.json
# Summary: 42/42 passed
```

Chi tiết 42 tests gồm 3 nhóm:
- [7 unit tests thuần](../../tests/unit/test_inference_pure.py): Cấu trúc dữ liệu `PaddingMetadata`, `InferenceResult.to_dict()`, công thức padding với các kích thước khác nhau, kiểm tra điều kiện biên kích thước quá nhỏ, phân cấp ngoại lệ. Hoàn toàn không cần PyTorch.
- [3 unit tests mock validation](../../tests/unit/test_inference_pure.py): Từ chối mock mode lạ, factor tùy biến và factor không hợp lệ.
- [10 pipeline integration tests](../../tests/unit/test_inference_pipeline.py): Xác thực tensor đầu vào, round-trip pad/unpad bảo toàn nội dung, các mode mock model (identity, invert, shift), từ chối kích thước không chia hết 32, chạy single với đo thời gian, hỗ trợ `InferenceRequest` và duck-typed `Sample`, chạy batch tuần tự bảo toàn thứ tự, và truyền lỗi model.
- [22 contract tests](../../tests/unit/test_inference_contracts.py): Kiểm tra chặt chẽ các ràng buộc C-01/C-02/C-04 (từ chối model ở `training` mode, bảo vệ device cố định của model, từ chối device sai, xử lý sample ID fallback, cô lập metadata, kiểm tra output dtype/shape/non-finite, batch rỗng, batch báo lỗi kèm index).

Toàn bộ 42 tests chạy độc lập trên CPU trong 0.04s mà không cần checkpoint weights ngoài Git hay GPU.

## Ranh giới bàn giao và bước tiếp theo

WS3 cung cấp `InferencePipeline` và `InferenceResult` để WS4 sử dụng trực tiếp trong module đánh giá chất lượng (C-05).
Task tiếp theo là **`M1-WS3-03`** sẽ xây dựng CLI wrapper và orchestration cho phép đọc ảnh từ file đĩa (PNG/JPG), chuyển đổi tensor và lưu ảnh kết quả vào thư mục `results/`.
