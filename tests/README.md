# Tests

M0 có test cho tooling không cần GPU. Các test model/checkpoint/inference chỉ được thêm khi implementation và fixtures hợp lệ tồn tại; chúng cần marker riêng và không chạy mặc định trên CI.

WS1-02 bổ sung `unit/test_model_spec.py` (standard library) và `model/test_model_adapter.py`.
Nhóm `model/` là opt-in qua `FFTFORMER_RUN_MODEL_TESTS=1`, cần PyTorch, einops và source upstream;
không dùng checkpoint/dataset.

WS1-03 thêm `model/test_checkpoint_loader.py`: 14 test dùng fixture Conv2d mock trong thư mục tạm,
3 integration tests chỉ chạy thêm khi `FFTFORMER_GOPRO_CHECKPOINT` được đặt.
`unit/test_ws1_03_verify.py` có 6 test không cần dependency để kiểm chứng reporter,
gồm subtest lỗi/skip và suite rỗng. CI mặc định hiện chạy 20 test nhẹ, skip 33 model/checkpoint tests.
Lệnh, marker và kết quả thực tế tại [model guide](../docs/modules/model.md#kiểm-thử-ws1-03-và-evidence-hiện-tại).
