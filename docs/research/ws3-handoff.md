# Bàn giao WS3 — Inference & Demonstration

Ngày kiểm chứng: 17/11/2026. Owner role: WS3. Phạm vi: `M1-WS3-02` theo [backlog](../backlog.md).

**Trạng thái công việc:** Task `M1-WS3-02` (Inference pipeline core) hoàn thành (`DONE`), đạt 42/42 CPU checks theo đúng chuẩn hợp đồng [C-02/C-04](../contracts.md). WS4 là consumer trực tiếp của `InferenceResult` phục vụ tính toán metric PSNR/SSIM. WS1 là provider của `ModelAdapter`. Phân công tại [ownership](../ownership.md), evidence tích hợp tại [project status](../project-status.md).

## Kết quả theo task

| Task | Yêu cầu và kết quả | Evidence |
| --- | --- | --- |
| M1-WS3-02 | Pipeline điều phối suy luận đơn lẻ (`run_single`) và batch (`run_batch`); tính toán reflection padding bội số 32 ở cạnh phải/đáy theo baseline; unpad khôi phục đúng kích thước gốc $H \times W$; clamp $[0, 1]$; `MockModel` với nhãn `is_mock = True`; xử lý ngoại lệ phân cấp rõ ràng; hỗ trợ duck-typed `Sample` từ C-03. | [API và tests](../modules/inference.md); [42/42 CPU checks](ws3-02-runtime-evidence.json); script xác minh `scripts/ws3_verify.py` |

## Files/modules bàn giao

| Nhóm | Files |
| --- | --- |
| WS3 inference API | `src/fftformer_lab/inference/types.py`, `padding.py`, `mock.py`, `pipeline.py`, `__init__.py`, `README.md` trong cùng module |
| WS3 tests | `tests/unit/test_inference_pure.py` (pure Python không cần PyTorch), `tests/unit/test_inference_pipeline.py` (PyTorch CPU mock pipeline), `tests/unit/test_inference_contracts.py` (ràng buộc Contract C-01/C-02/C-04) |
| Script & evidence | `scripts/ws3_verify.py`, `docs/research/ws3-02-runtime-evidence.json` |
| Hướng dẫn sử dụng & điều phối | `docs/modules/inference.md`, `scripts/README.md`, `tests/README.md`, `docs/project-status.md`, `docs/contribution-log.md`, `docs/backlog.md`, `docs/ownership.md`, `docs/research/ws3-handoff.md` |

Thay đổi shared contract: **Không**. Triển khai public Python API theo đúng [C-02/C-04](../contracts.md): `InferencePipeline`, `InferenceRequest`, `InferenceResult`, `PaddingMetadata`, `MockModel`, `calculate_reflection_padding`, `pad_image`, `unpad_image`, `validate_input_tensor` và các ngoại lệ phân cấp (`InferenceError`, `InferenceInputError`, `InferencePaddingError`, `InferenceExecutionError`).

## Ranh giới bàn giao cho Consumer (WS4)

- WS4 nhận `InferenceResult` cho từng sample:
  - `restored_image`: tensor RGB float32 CHW `[3, H, W]` đã được unpad/crop về đúng kích thước gốc $H \times W$ và clamp trong đoạn $[0.0, 1.0]$.
  - `sample_id`: định danh của mẫu ảnh, bảo toàn từ input request / dataset sample.
  - `input_shape`, `padded_shape`, `output_shape`: bộ ba kích thước phục vụ kiểm chứng padding.
  - `padding`: đối tượng `PaddingMetadata` ghi nhận `(top, bottom, left, right)` và `mode="reflect"`.
  - `device`: thiết bị thực tế đã thực thi suy luận (ví dụ `"cpu"`).
  - `elapsed_time_seconds`: thời gian thực thi model tính bằng giây (chính xác với `time.perf_counter()`, có đồng bộ CUDA nếu chạy GPU).
  - `is_mock`: cờ boolean xác định rõ ràng kết quả đến từ `MockModel` hay model thật.
  - `model_metadata`: bản sao metadata từ model provider.
- `run_batch` xử lý tuần tự từng sample để bảo toàn metadata độc lập và thứ tự duyệt của dataset; không tự ý gộp tensor thành batch NCHW lớn gây mất dấu sample metadata.

## Commands và kết quả thực tế

Tại repository root trên môi trường Windows, Python 3.11.9, PyTorch 2.15.0+cpu:

```powershell
python -B scripts/ws3_verify.py --repo . --output docs/research/ws3-02-runtime-evidence.json
# Ran 42 tests in 0.035s; OK (42/42 passed); xuất bằng chứng docs/research/ws3-02-runtime-evidence.json

python -B -m unittest discover -s tests -p "test_inference_*.py" -v
# 42/42 tests PASS trên CPU

python -m compileall -q scripts src tests
# Exit 0, không có lỗi cú pháp
```

## Hạn chế đã công bố

- `UNVERIFIED`: CUDA inference, nạp checkpoint thật trên GPU, benchmark chất lượng PSNR/SSIM, và training.
- File-based image I/O CLI (đọc ghi file `.png`/`.jpg` từ đĩa qua OpenCV/Pillow) thuộc phạm vi của task `M1-WS3-03` tiếp theo.
- Hiện tại các bài test sử dụng `MockModel` (identity/invert/shift) để đảm bảo tính độc lập và khả năng chạy kiểm thử 100% trên CPU không phụ thuộc GPU hay weights ngoài Git.
