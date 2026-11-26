# Bàn giao WS3 — Inference & Demonstration

Ngày kiểm chứng: 18/11/2026. Owner role: WS3. Phạm vi: `M1-WS3-02` và `M1-WS3-03` theo [backlog](../backlog.md).

**Trạng thái công việc:** Toàn bộ 2 task Wave C của WS3 (`M1-WS3-02` inference pipeline core và `M1-WS3-03` CLI/orchestration) đã hoàn thành (`DONE`), đạt 57/57 CPU checks theo đúng chuẩn hợp đồng [C-01/C-02/C-04](../contracts.md). WS4 là consumer trực tiếp của `InferenceResult` phục vụ tính toán metric PSNR/SSIM. WS1 là provider của `ModelAdapter`, WS2 là provider của `read_image`, `ImageDataset` và manifest. Phân công tại [ownership](../ownership.md), evidence tích hợp tại [project status](../project-status.md).

## Kết quả theo task

| Task | Yêu cầu và kết quả | Evidence |
| --- | --- | --- |
| M1-WS3-02 | Pipeline điều phối suy luận đơn lẻ (`run_single`) và batch (`run_batch`); tính toán reflection padding bội số 32 ở cạnh phải/đáy theo baseline; unpad khôi phục đúng kích thước gốc $H \times W$; clamp $[0, 1]$; `MockModel` với nhãn `is_mock = True`; xử lý ngoại lệ phân cấp rõ ràng; hỗ trợ duck-typed `Sample` từ C-03. | [API và tests](../modules/inference.md); 42 checks CPU ban đầu |
| M1-WS3-03 | CLI orchestration (`cli.py`, `__main__.py`); serialization ảnh đầu ra `write_image` tuân thủ C-02 (clamp $[0, 1]$, lượng tử hóa uint8); hỗ trợ ảnh đơn lẻ (`--input`), thư mục ảnh (`--input-dir`) và manifest của WS2 (`--manifest`); bảo toàn thứ tự batch; tương thích với `read_image`, `ImageDataset` từ WS2 và `load_model` từ WS1; xuất metadata JSON tùy chọn (`--save-metadata`). | [15 CLI & IO tests](../../tests/unit/test_inference_cli.py); [57/57 CPU checks](ws3-02-runtime-evidence.json); script xác minh `scripts/ws3_verify.py` |

## Files/modules bàn giao

| Nhóm | Files |
| --- | --- |
| WS3 inference API | `src/fftformer_lab/inference/types.py`, `padding.py`, `mock.py`, `pipeline.py`, `io.py`, `cli.py`, `__main__.py`, `__init__.py`, `README.md` trong cùng module |
| WS3 tests | `tests/unit/test_inference_pure.py` (pure Python không cần PyTorch), `tests/unit/test_inference_pipeline.py` (PyTorch CPU mock pipeline), `tests/unit/test_inference_contracts.py` (ràng buộc Contract C-01/C-02/C-04), `tests/unit/test_inference_cli.py` (CLI & Image IO tests) |
| Script & evidence | `scripts/ws3_verify.py`, `docs/research/ws3-02-runtime-evidence.json` |
| Hướng dẫn sử dụng & điều phối | `docs/modules/inference.md`, `scripts/README.md`, `tests/README.md`, `docs/project-status.md`, `docs/contribution-log.md`, `docs/backlog.md`, `docs/ownership.md`, `docs/research/ws3-handoff.md` |

Thay đổi shared contract: **Không**. Triển khai public Python API theo đúng [C-01/C-02/C-04](../contracts.md): `InferencePipeline`, `InferenceRequest`, `InferenceResult`, `PaddingMetadata`, `MockModel`, `calculate_reflection_padding`, `pad_image`, `unpad_image`, `write_image`, `run_cli`, `validate_input_tensor` và các ngoại lệ phân cấp (`InferenceError`, `InferenceInputError`, `InferencePaddingError`, `InferenceExecutionError`, `InferenceIOError`).

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
- Hỗ trợ lưu ảnh phục hồi ra đĩa thông qua `write_image(tensor, path)`.

## Commands và kết quả thực tế

Tại repository root trên môi trường Windows, Python 3.11.9, PyTorch 2.15.0+cpu:

```powershell
python -B scripts/ws3_verify.py --repo . --output docs/research/ws3-02-runtime-evidence.json
# Ran 57 tests in 0.234s; OK (57/57 passed); xuất bằng chứng docs/research/ws3-02-runtime-evidence.json

python -B -m unittest discover -s tests -p "test_inference_*.py" -v
# 57/57 tests PASS trên CPU

python -B -m unittest discover -s tests -p "test_*.py"
# 152 tests: 119 PASS trên CPU, 33 opt-in tests skip; không có lỗi

python -m compileall -q scripts src tests
# Exit 0, không có lỗi cú pháp
```

## Hạn chế đã công bố

- Bằng chứng bàn giao WS3 ghi lại bộ ca CPU. Lượt kiểm chứng sau đó đã chạy CLI với checkpoint GoPro thật trên CUDA. Benchmark PSNR/SSIM toàn tập vẫn `UNVERIFIED`. [Trạng thái dự án](../project-status.md)
- Mặc định CLI sử dụng `MockModel` khi không truyền đường dẫn checkpoint; khi muốn chạy checkpoint thật cần chỉ định `--checkpoint` và `--architecture-file` hợp lệ.
