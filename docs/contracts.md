# API contracts liên module

**Trạng thái: `APPROVED` — M1.0 contract baseline v1.0, quyết định bởi Tech Lead.** Đây là specification đã chốt cho group implementation, **không phải** bằng chứng API Python, checkpoint compatibility, CPU inference hay benchmark đã được chạy. Evidence từ upstream và các lựa chọn do Tech Lead quyết định được ghi trong [ADR-0002](decisions/0002-m1-contract-baseline.md). Mọi thay đổi cross-module sau M1.0 cần task được Tech Lead cho phép và cập nhật ADR/contract tương ứng.

## C-01 Model loading

- **Status:** `APPROVED`.
- **Provider → consumer:** WS1 → WS3, WS4.
- **Input:** `ModelSpec` gồm architecture identifier/version, checkpoint path, device request và strictness (mặc định đề xuất `strict=True`, cấm tự chuyển `False` để che mismatch).
- **Output:** model ở evaluation mode cùng metadata: resolved device, upstream/group implementation id + commit pin, checkpoint identity (SHA-256 nếu đã tính). Group adapter phải resolve `map_location` theo device đã chọn; đây là quyết định portability của nhóm, không phải hành vi `test.py` upstream.
- **Preconditions:** architecture/checkpoint là cặp tương thích đã xác minh; checkpoint path tồn tại; device khả dụng.
- **Postconditions:** model không bị mutate ngoài `eval()`; incompatibility được báo rõ, không load im lặng với `strict=False`.
- **Errors:** file thiếu, checkpoint schema mismatch, state-dict key/shape mismatch, unsupported device.
- **Test chấp nhận:** fixture checkpoint nhỏ được phép phân phối, hoặc marker integration có artifact ngoài Git. `PLANNED`.

## C-02 Image input/output

- **Status:** `APPROVED`.
- **Provider → consumer:** WS2 → WS1, WS3, WS4.
- **Input chuẩn đề xuất:** ảnh màu **RGB**; một ảnh ở boundary đọc/ghi dùng tensor `float32` `CHW` (`[3,H,W]`), model nhận `NCHW` (`[N,3,H,W]`), range `[0,1]`; chuyển đổi `CHW→NCHW` phải explicit. Boundary đọc file chịu trách nhiệm BGR→RGB nếu dùng OpenCV. Không ngầm định ảnh grayscale/alpha tự động tương thích.
- **Output:** boundary đọc/ghi trả tensor RGB `float32` `CHW` (`[3,H,W]`) có spatial extent `H×W` gốc sau khi loại padding; model-facing output dùng `NCHW` (`[N,3,H,W]`). Serialize phải nêu format, quantization và chính sách clamp.
- **Preconditions:** finite values, 3 channels, dimension batch rõ ràng. Padding phải ghi metadata `(top, bottom, left, right)`; baseline official hiện pad phải/dưới bằng reflection tới bội số 32. Trường hợp ảnh quá nhỏ để reflection padding phải trả lỗi rõ hoặc có policy được duyệt riêng (không âm thầm đổi baseline).
- **Errors:** non-RGB không có conversion rõ ràng, shape/range không hợp lệ, ảnh quá nhỏ cho reflection padding, out-of-memory.
- **Test chấp nhận:** round-trip conversion/range/shape và input invalid. `PLANNED`.

## C-03 Dataset

- **Status:** `APPROVED`.
- **Provider → consumer:** WS2 → WS3, WS4.
- **Input:** manifest hoặc root có cặp `input`/`target` cùng tên khi có ground truth; layout `test/input` + `test/target` là upstream test convention đã audit, không bắt buộc mọi dataset phải theo layout đó. Ảnh đơn lẻ phải được đánh dấu `has_ground_truth=false`.
- **Output:** `Sample` với `image: float32 CHW [3,H,W]`, `target: float32 CHW [3,H,W] | None`, `sample_id`, `source_path`, preprocessing metadata và `has_ground_truth`.
- **Preconditions:** cặp phải match 1:1 theo sample ID và thứ tự duyệt deterministic khi yêu cầu supervision; kiểm tra kích thước và color space; policy crop/resize/normalization được version trong config.
- **Postconditions:** loader không thay đổi file nguồn và không bịa target.
- **Errors:** root thiếu, extension không hỗ trợ, pair thiếu/trùng, mismatch size/channel.
- **Test chấp nhận:** fixture tối thiểu dùng ảnh hợp pháp do nhóm tạo. `PLANNED`.

## C-04 Inference result

- **Status:** `APPROVED`.
- **Provider → consumer:** WS3 → WS4 (và caller/demo sau milestone phù hợp).
- **Input:** model đã load và `Sample` hoặc single image theo C-02.
- **Output:** một `InferenceResult` cho một `Sample`, gồm `restored_image: float32 CHW [3,H,W]` đã crop về kích thước gốc, `sample_id`, input/output shape, padding metadata, model/checkpoint id, device thực dùng và elapsed time chỉ khi thật sự đo (đơn vị đo explicit). Batch orchestration trả sequence theo thứ tự sample, không ngầm ghép output thành batch mới.
- **Preconditions:** model và input tuân C-01/C-02. Ảnh không GT vẫn hợp lệ.
- **Errors:** propagate có ngữ cảnh cho load/device/shape/output-write failure; không che lỗi bằng output giả.
- **Test chấp nhận:** smoke test CPU hoặc GPU được ghi marker; không tuyên bố support CPU khi chưa xác minh. `PLANNED`.

## C-05 Evaluation result

- **Status:** `APPROVED`.
- **Provider → consumer:** WS4 → experiment metadata và các workstream cần đối chiếu kết quả; WS2/WS3 cung cấp sample/prediction đầu vào.
- **Input:** prediction và GT đã được xác nhận alignment, protocol identifier có version/source, metric configuration (bao gồm color space/range/crop/quantization). Mỗi metric được tính theo từng sample; NCHW batch phải được tách thành sample trước khi gọi evaluator upstream. Với profile `upstream-basicsr` dùng nhánh `use_image=True`, adapter WS4 phải chuyển từng tensor RGB `CHW`/`NCHW` qua `tensor2img` thành `ndarray` HWC trước khi gọi metric, chọn và ghi rõ `rgb2bgr`, rồi áp dụng cùng representation cho prediction và GT. Không truyền NCHW trực tiếp cho `calculate_psnr`/`calculate_ssim`.
- **Output:** `EvaluationResult` gồm PSNR/SSIM (nếu tính và theo protocol đã xác minh), số sample hợp lệ/bị loại, crop/color/range/quantization policy, commit/config/checkpoint metadata; phân biệt `paper-reported`, `group-measured` và `UNVERIFIED`.
- **Preconditions:** GT tồn tại; cùng shape/colorspace/range theo protocol. Không tính metric cho input không GT.
- **Errors:** missing GT, shape mismatch, unknown protocol, non-finite metric input.
- **Test chấp nhận:** deterministic fixture + reference calculation; đối chiếu official protocol trước khi gắn nhãn reproduction. `PLANNED`.

**Profile upstream đã audit:** YAML GoPro/RealBlur đặt `crop_border: 0`, `test_y_channel: false`. `calculate_psnr`/`calculate_ssim` khai báo `input_order` `HWC` hoặc `CHW`, nhưng nhánh validation `use_image=True` của upstream gọi `tensor2img` trước; hàm đó nhận tensor RGB và, khi `rgb2bgr=True`, trả `ndarray` HWC BGR `uint8`. Vì vậy adapter phải dùng representation/profile được ghi nhận rõ, không suy diễn rằng mọi tensor CHW/NCHW được metric xử lý nguyên trạng. `calculate_ssim` mặc định chọn đường `ssim3d=True` có CUDA call, do đó không được coi là CPU-safe. `realblur-official` là protocol riêng cho claim RealBlur và không được thay thế bằng profile này.

## C-06 Configuration

- **Status:** `APPROVED`.
- **Provider → consumer:** WS4 → WS1, WS2, WS3.
- **Input:** file config versioned với schema/version, dataset/checkpoint reference và expected contract version; đường dẫn cục bộ/remote chỉ được resolve lúc runtime (Windows/Linux) thay vì hardcode `C:\` hoặc `/home/...`.
- **Output:** immutable resolved configuration được ghi trong experiment metadata.
- **Preconditions:** không chứa secret/path cá nhân hard-code; đường dẫn được validate tại runtime.
- **Errors:** key thiếu, schema version không hỗ trợ, file/path không tồn tại.
- **Test chấp nhận:** parse + validation test không phụ thuộc GPU. `PLANNED`.

## Ma trận review và quy tắc freeze (M1.0)

| Contract | Owner role | Consumer role | Freeze / change gate |
| --- | --- | --- | --- |
| C-01 | WS1 | WS3 | Tech Lead approval; strict load + metadata + checkpoint compatibility plan |
| C-02 | WS2 | WS1, WS3 | Tech Lead approval; RGB/layout/range/padding & invalid-input policy |
| C-03 | WS2 | WS4 | Tech Lead approval; pairing/manifest/GT policy + fixture plan |
| C-04 | WS3 | WS2, WS4 | Tech Lead approval; shape restoration, device, errors & smoke-test plan |
| C-05 | WS4 | WS2 | Tech Lead approval; per-dataset metric protocol + GT/crop policy; chưa claim RealBlur reproduction |
| C-06 | WS4 | WS1, WS2, WS3 | Tech Lead approval; schema version, paths, experiment metadata & config validation |

Đây là vai trò workstream, **không phải tên thành viên đã được chỉ định**. Tech Lead đã đóng băng interface v1.0; feedback kỹ thuật của consumer vẫn phải được ghi trong task/PR, nhưng không tự mở rộng scope hay sửa contract. Mỗi implementation task phải nằm trong phạm vi Tech Lead đã cho phép.
