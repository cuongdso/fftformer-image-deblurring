# WS4 — Evaluation, Reproducibility & Integration

**Trạng thái:** `M1-WS4-02` đã triển khai Evaluation Engine cho Contract C-05. Đây là đo lường cấp nhóm (*group-measured*), không phải tái hiện benchmark hay kết quả báo cáo trong paper.

## Phạm vi hiện có

Module `fftformer_lab.evaluation` nhận `Sample` từ WS2 (C-03) và `InferenceResult` từ WS3 (C-04), kiểm tra chúng cùng `sample_id`, cùng shape và tuân RGB `float32` CHW `[0,1]` (C-02), rồi đo PSNR/SSIM theo từng mẫu. API công khai gồm:

- `evaluate_sample(result, sample, protocol, *, basicsr_adapter=None)`
- `evaluate_batch(results, samples, protocol, *, basicsr_adapter=None)`
- `MetricProtocol`, `SampleEvaluation`, `EvaluationSummary`
- `EvaluationInputError`, `UnknownProtocolError`, `ProtocolUnavailableError`

Batch giữ nguyên thứ tự `InferenceResult`; mẫu thiếu GT được loại trừ có lý do `missing-ground-truth`, không bị bỏ qua im lặng. Metadata kết quả lưu profile protocol, representation, conversion và `model_metadata` nếu WS3 cung cấp.

Package protocol/error vẫn import được trong lightweight CI không có dependency computer-vision. Chỉ khi gọi conversion hoặc metric thì engine mới yêu cầu NumPy; `group-basicsr-cpu` SSIM còn cần OpenCV và báo `ProtocolUnavailableError` kèm hướng dẫn cài đặt thay vì lỗi import. Test metric được skip có lý do trong môi trường đó.

## Protocol

- `upstream-basicsr`: HWC BGR `uint8` sau clamp/round như `tensor2img`; `ssim3d=True` và cần adapter CUDA tường minh. Không có fallback CPU.
- `group-basicsr-cpu`: cùng conversion/PSNR, nhưng dùng SSIM Gaussian 2D tương ứng nhánh `ssim3d=False`; là protocol riêng, không dùng để khẳng định benchmark upstream.
- `realblur-official`: đã đăng ký để không bị nhầm với BasicSR, nhưng chủ đích trả `ProtocolUnavailableError` vì ECC alignment và evaluator file-based chính thức chưa thuộc WS4-02.

Chi tiết kiến trúc và các giới hạn được giải thích trong [Evaluation Engine](../ws4/evaluation-engine.md) và [Metric protocols](../ws4/metric-protocols.md).

## WS4-03: config, metadata và CLI

`ResolvedEvaluationConfig` đọc JSON schema v1 theo Contract C-06, từ chối key lạ và resolve mọi path tương đối theo thư mục config. `build_experiment_metadata`/`write_experiment_metadata` lưu provenance và C-05 summary bằng JSON strict. CLI `python -m fftformer_lab.evaluation.cli validate-config <config>` không cần dependency CV; `run <config>` chỉ đánh giá prediction images đã có sẵn với GT qua WS2/WS4, không chạy inference hay gán nguồn FFTformer không có evidence.

## Chưa thuộc WS4-03

WS4-04 vẫn sở hữu integration gate giữa public adapters. Chưa chạy benchmark GoPro/RealBlur, không có checkpoint/dataset result và chưa xác minh BasicSR SSIM CUDA hay RealBlur official pipeline trong repository này.
