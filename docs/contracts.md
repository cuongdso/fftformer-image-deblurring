# API contracts liên module

**Trạng thái toàn bộ tài liệu: `PROPOSED` (M0).** Đây là design boundary, không phải API đã triển khai hoặc đã duyệt. Mỗi contract chỉ chuyển sang `APPROVED` sau issue/PR có owner WS1–WS4 và test tương ứng.

## C-01 Model loading

- **Input:** `ModelSpec` gồm architecture identifier, checkpoint path, device request và strictness.
- **Output:** model ở evaluation mode cùng metadata: resolved device, upstream/group implementation id, checkpoint checksum nếu có.
- **Preconditions:** architecture/checkpoint là cặp tương thích đã xác minh; checkpoint path tồn tại; device khả dụng.
- **Postconditions:** model không bị mutate ngoài `eval()`; incompatibility được báo rõ, không load im lặng với `strict=False`.
- **Errors:** file thiếu, checkpoint schema mismatch, state-dict key/shape mismatch, unsupported device.
- **Test chấp nhận:** fixture checkpoint nhỏ được phép phân phối, hoặc marker integration có artifact ngoài Git. `PLANNED`.

## C-02 Image input/output

- **Input chuẩn đề xuất:** ảnh màu **RGB**, tensor `float32` dạng `CHW` hoặc batch `NCHW`, range `[0, 1]`; boundary đọc file chịu trách nhiệm BGR→RGB nếu dùng OpenCV.
- **Output:** tensor RGB `float32` cùng spatial extent ảnh gốc sau khi loại padding; serialize ảnh phải nêu format và quantization.
- **Preconditions:** finite values, 3 channels, dimension batch rõ ràng. Padding phải ghi metadata `(top, bottom, left, right)`; baseline official hiện pad phải/dưới reflection tới bội số 32.
- **Errors:** non-RGB không có conversion rõ ràng, shape/range không hợp lệ, ảnh quá nhỏ cho reflection padding, out-of-memory.
- **Test chấp nhận:** round-trip conversion/range/shape và input invalid. `PLANNED`.

## C-03 Dataset

- **Input:** manifest hoặc root có cặp `input`/`target` cùng tên (khi có ground truth); ảnh đơn lẻ phải được đánh dấu `has_ground_truth=false`.
- **Output:** `Sample` với `image`, `target | None`, `sample_id`, `source_path`, preprocessing metadata.
- **Preconditions:** cặp phải match 1:1 khi yêu cầu supervision; policy crop/resize/normalization được version trong config.
- **Postconditions:** loader không thay đổi file nguồn và không bịa target.
- **Errors:** root thiếu, extension không hỗ trợ, pair thiếu/trùng, mismatch size/channel.
- **Test chấp nhận:** fixture tối thiểu dùng ảnh hợp pháp do nhóm tạo. `PLANNED`.

## C-04 Inference result

- **Input:** model đã load và `Sample` hoặc single image theo C-02.
- **Output:** `InferenceResult` gồm restored image, `sample_id`, input/output shape, padding metadata, model/checkpoint id, elapsed time nếu thật sự đo.
- **Preconditions:** model và input tuân C-01/C-02. Ảnh không GT vẫn hợp lệ.
- **Errors:** propagate có ngữ cảnh cho load/device/shape/output-write failure; không che lỗi bằng output giả.
- **Test chấp nhận:** smoke test CPU hoặc GPU được ghi marker; không tuyên bố support CPU khi chưa xác minh. `PLANNED`.

## C-05 Evaluation result

- **Input:** prediction và GT đã được xác nhận alignment, protocol identifier, metric configuration.
- **Output:** `EvaluationResult` gồm PSNR/SSIM (nếu tính), số sample, crop/color/range policy, commit/config/checkpoint metadata.
- **Preconditions:** GT tồn tại; cùng shape/colorspace/range theo protocol. Không tính metric cho input không GT.
- **Errors:** missing GT, shape mismatch, unknown protocol, non-finite metric input.
- **Test chấp nhận:** deterministic fixture + reference calculation; đối chiếu official protocol trước khi gắn nhãn reproduction. `PLANNED`.

## C-06 Configuration

- **Input:** file config versioned với schema/version, owner, dataset/checkpoint reference và expected contract version.
- **Output:** immutable resolved configuration được ghi trong experiment metadata.
- **Preconditions:** không chứa secret/path cá nhân hard-code; đường dẫn được validate tại runtime.
- **Errors:** key thiếu, schema version không hỗ trợ, file/path không tồn tại.
- **Test chấp nhận:** parse + validation test không phụ thuộc GPU. `PLANNED`.
