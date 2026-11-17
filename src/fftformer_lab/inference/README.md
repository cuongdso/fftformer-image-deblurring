# WS3 — Inference & Demonstration

Module phục vụ thực thi suy luận khử mờ ảnh đơn lẻ và theo batch (Contract C-04, C-02).

## Cấu trúc module

```text
src/fftformer_lab/inference/
├── __init__.py        # Public API exports
├── types.py           # Data structures (InferenceRequest, InferenceResult, PaddingMetadata) và exceptions
├── padding.py         # Tính toán và thực hiện reflection padding bội 32, crop phục hồi HxW gốc
├── mock.py            # MockModel test double phục vụ kiểm thử không cần GPU/checkpoint nặng
├── pipeline.py        # InferencePipeline điều phối tiền xử lý, forward mô hình và hậu xử lý
└── README.md
```

## Các thành phần chính

1. **`InferencePipeline`**:
   - Nhận vào `ModelAdapter` hoặc `MockModel` (hoặc callable phù hợp C-01/C-02).
   - `run_single(request)`: Xác minh tensor RGB float32 `[3, H, W]` trong khoảng `[0, 1]`, reflection pad tới bội số 32, chuyển batch `[1, 3, H_pad, W_pad]`, forward không tính đạo hàm (`torch.no_grad()`), đo thời gian chạy (giây), unpad về kích thước gốc $H \times W$, clamp `[0, 1]` và trả về `InferenceResult`.
   - `run_batch(requests)`: Điều phối suy luận tuần tự theo thứ tự mẫu, giữ nguyên sample ID và metadata.
2. **`PaddingMetadata` & `calculate_reflection_padding`**:
   - Quy ước pad cạnh phải và đáy bằng reflection tới bội số của 32 (`top=0, bottom=h_n, left=0, right=w_n`).
   - Kiểm tra điều kiện biên: nếu kích thước ảnh nhỏ hơn hoặc bằng lượng pad cần thiết, trả lỗi `InferencePaddingError` rõ ràng.
3. **`MockModel`**:
   - Hỗ trợ các mode: `identity`, `invert`, `shift`.
   - Đánh dấu tường minh `is_mock = True`, không giả mạo là kết quả FFTformer thật.
