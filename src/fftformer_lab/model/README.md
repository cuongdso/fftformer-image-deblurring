# WS1 — Model & Architecture

`M1-WS1-02` có `ModelConfig`, `ModelSpec`, `ModelMetadata`, `ModelAdapter` và `build_model`.
API, ví dụ sử dụng và kết quả kiểm thử tại [module guide](../../../docs/modules/model.md).

Adapter nhận đường dẫn source upstream tường minh và chỉ khởi tạo model chưa nạp trọng số.
Checkpoint loader thuộc `M1-WS1-03`; `build_model` báo `NotImplementedError` khi có yêu cầu checkpoint.
