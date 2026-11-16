# WS1 — Model & Architecture

`M1-WS1-02` có `ModelConfig`, `ModelSpec`, `ModelMetadata`, `ModelAdapter` và `build_model`.
`M1-WS1-03` bổ sung `load_model` và các lỗi checkpoint có ngữ cảnh.
API, ví dụ sử dụng và kết quả kiểm thử tại [module guide](../../../docs/modules/model.md).

Adapter nhận đường dẫn source upstream tường minh và chỉ khởi tạo model chưa nạp trọng số.
Để nạp checkpoint, gọi `load_model` với `ModelSpec.checkpoint` tường minh;
`build_model` vẫn báo `NotImplementedError` khi có yêu cầu checkpoint.
