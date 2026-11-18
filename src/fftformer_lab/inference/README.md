# WS3 — Inference & Demonstration

Module phục vụ thực thi suy luận khử mờ ảnh đơn lẻ, theo thư mục hoặc manifest (Contract C-04, C-02).

## Cấu trúc module

```text
src/fftformer_lab/inference/
├── __init__.py        # Public API exports
├── types.py           # Data structures (InferenceRequest, InferenceResult, PaddingMetadata) và exceptions
├── padding.py         # Tính toán và thực hiện reflection padding bội 32, crop phục hồi HxW gốc
├── mock.py            # MockModel test double phục vụ kiểm thử không cần GPU/checkpoint nặng
├── pipeline.py        # InferencePipeline điều phối tiền xử lý, forward mô hình và hậu xử lý
├── io.py              # Serialization và lưu ảnh output (PNG, JPEG) theo C-02
├── cli.py             # CLI orchestrator cho ảnh đơn, thư mục và dataset manifest
├── __main__.py        # Entry point cho lệnh `python -m fftformer_lab.inference`
└── README.md
```

## Các thành phần chính

1. **`InferencePipeline`**:
   - Nhận vào `ModelAdapter` hoặc `MockModel` (hoặc callable phù hợp C-01/C-02).
   - `run_single(request)`: Xác minh tensor RGB float32 `[3, H, W]` trong khoảng `[0, 1]`, reflection pad tới bội số 32, chuyển batch `[1, 3, H_pad, W_pad]`, forward không tính đạo hàm (`torch.no_grad()`), đo thời gian chạy (giây), unpad về kích thước gốc $H \times W$, clamp `[0, 1]` và trả về `InferenceResult`.
   - `run_batch(requests)`: Điều phối suy luận tuần tự theo thứ tự mẫu, giữ nguyên sample ID và metadata.
2. **`write_image`** (`io.py`):
   - Lưu tensor RGB float32 `[3, H, W]` thành file ảnh 8-bit PNG hoặc JPEG.
   - Tuân thủ Contract C-02: clamp $[0.0, 1.0]$, lượng tử hóa `round(x * 255.0) -> uint8`.
3. **CLI Orchestration** (`cli.py`):
   - Chạy trên một ảnh đơn: `python -m fftformer_lab.inference -i anh_mo.png -o ket_qua.png --mock`
   - Chạy trên thư mục ảnh: `python -m fftformer_lab.inference --input-dir thu_muc_in/ --output-dir thu_muc_out/ --mock`
   - Chạy qua manifest WS2: `python -m fftformer_lab.inference -m manifest.json --output-dir thu_muc_out/ --mock`
   - Hỗ trợ lưu metadata kèm theo (`--save-metadata`).
4. **`MockModel`**:
   - Hỗ trợ các mode: `identity`, `invert`, `shift`.
   - Đánh dấu tường minh `is_mock = True`, không giả mạo là kết quả FFTformer thật.
