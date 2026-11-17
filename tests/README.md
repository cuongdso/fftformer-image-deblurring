# Tests

M0 có test cho tooling không cần GPU. Các test model/checkpoint/inference chỉ được thêm khi implementation và fixtures hợp lệ tồn tại; chúng cần marker riêng và không chạy mặc định trên CI.

WS1-02 bổ sung `unit/test_model_spec.py` (standard library) và `model/test_model_adapter.py`.
Nhóm `model/` là opt-in qua `FFTFORMER_RUN_MODEL_TESTS=1`, cần PyTorch, einops và source upstream;
không dùng checkpoint/dataset.

WS1-03 thêm `model/test_checkpoint_loader.py`: 14 test dùng fixture Conv2d mock trong thư mục tạm,
3 integration tests chỉ chạy thêm khi `FFTFORMER_GOPRO_CHECKPOINT` được đặt.
`unit/test_ws1_03_verify.py` có 8 test không cần dependency để kiểm chứng reporter,
gồm subtest lỗi/skip, nhãn trùng và suite rỗng. CI mặc định hiện chạy 22 test nhẹ, skip 33 model/checkpoint tests.
Lệnh, marker và kết quả thực tế tại [model guide](../docs/modules/model.md#kiểm-thử-ws1-03-và-evidence-hiện-tại).


## WS3 Tests
WS3-02 bổ sung 42 tests cho inference pipeline theo Contract C-02/C-04:

- `unit/test_inference_pure.py`: 10 unit tests độc lập không cần PyTorch (PaddingMetadata, công thức padding, kiểm tra điều kiện biên kích thước quá nhỏ, phân cấp ngoại lệ).
- `unit/test_inference_pipeline.py`: 10 tests kiểm tra pipeline với `MockModel`, round-trip pad/unpad, batch sequence và truyền lỗi.
- `unit/test_inference_contracts.py`: 22 tests kiểm tra việc thực thi các ràng buộc hợp đồng C-01/C-02/C-04 (chặn training mode, bảo vệ device cố định, kiểm tra output shape/dtype/finiteness, batch error index).

Toàn bộ 42 tests WS3 chạy độc lập trên CPU mà không cần GPU hay checkpoint ngoài Git.
Lệnh và kết quả thực tế tại [inference guide](../docs/modules/inference.md#kiểm-thử-ws3-02-và-evidence-hiện-tại) và [ws3-02-runtime-evidence.json](../docs/research/ws3-02-runtime-evidence.json).
