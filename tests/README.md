# Tests

M0 có test cho tooling không cần GPU. Các test model/checkpoint/inference chỉ được thêm khi implementation và fixtures hợp lệ tồn tại; chúng cần marker riêng và không chạy mặc định trên CI.

WS1-02 bổ sung `unit/test_model_spec.py` (standard library) và `model/test_model_adapter.py`.
Nhóm `model/` là opt-in qua `FFTFORMER_RUN_MODEL_TESTS=1`, cần PyTorch, einops và source upstream;
không dùng checkpoint/dataset. Lệnh và kết quả thực tế tại [model guide](../docs/modules/model.md#kiểm-thử-đã-chạy).
