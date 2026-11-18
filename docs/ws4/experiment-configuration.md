# Cấu hình thí nghiệm và CLI đánh giá

## Mục đích

WS4-03 tạo ranh giới C-06 cho một lần đánh giá ảnh đã có sẵn. Cấu hình không chứa trọng số, dataset hay secret; nó chỉ định danh experiment, protocol, hai thư mục prediction/ground truth và nơi ghi metadata. Vì vậy một kết quả luôn có thể truy về policy metric và đường dẫn đầu vào đã dùng.

## Thành phần

`config.py` dùng `json`, `pathlib` và dataclass bất biến để đọc schema version 1. Validator từ chối key lạ, schema/contract/protocol không hợp lệ và thư mục không tồn tại. Path tương đối được resolve từ thư mục chứa config, không từ current working directory, nên cùng config hoạt động nhất quán trên Windows và Linux.

`metadata.py` chuyển `ResolvedEvaluationConfig` cùng `EvaluationSummary` thành JSON strict (`allow_nan=False`). Metadata ghi protocol, conversion/crop/quantization policy, identity dataset/model/checkpoint khi caller đã cung cấp, và để `null` khi chưa biết. PSNR vô cực vẫn dùng chuỗi `"inf"` theo WS4-02.

`cli.py` cung cấp hai lệnh:

```text
python -m fftformer_lab.evaluation.cli validate-config configs/examples/evaluation-precomputed.json
python -m fftformer_lab.evaluation.cli run path/to/job.json
```

`validate-config` chỉ kiểm C-06 và không cần Torch, NumPy, Pillow hay OpenCV. `run` dùng `build_paired_manifest`/`ImageDataset` của WS2 để ghép prediction và GT, rồi chuyển prediction thành adapter WS4 tối thiểu trước khi gọi Evaluation Engine. Adapter này biểu diễn ảnh prediction đã có sẵn, không phải `InferenceResult` và không tuyên bố ảnh do FFTformer sinh ra.

## Schema và quy trình

Schema `configs/evaluation.schema.json` yêu cầu `schema_version`, `contract_version`, `experiment_id`, `protocol_id`, `dataset`, `output_path`, cùng ba vùng provenance tùy chọn (`checkpoint`, `model`, `provenance`). Ví dụ config nằm ở `configs/examples/evaluation-precomputed.json`; trước khi chạy thật, người dùng thay các thư mục bằng paired RGB PNG/JPEG hợp lệ của mình.

Khi chạy, CLI kiểm output chưa tồn tại, ghép tên file prediction/GT theo WS2, đọc ảnh RGB, đánh giá theo protocol đã chọn, rồi ghi một JSON metadata. Sample thiếu hoặc mismatch không được thay bằng mock; lỗi protocol/runtime được trả về rõ ràng. `--overwrite` là lựa chọn tường minh để thay kết quả cũ.

## Giới hạn tái lập

Một file metadata ghi nhận evaluation của prediction images, không chứng minh inference FFTformer, checkpoint compatibility hay benchmark. `group-basicsr-cpu` cần NumPy/OpenCV; `upstream-basicsr` cần adapter CUDA tường minh; `realblur-official` chưa khả dụng ở WS4-03. WS4-03 không triển khai inference CLI, training, ECC alignment hoặc WS4-04 integration.
