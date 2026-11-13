# M1.0 — Interface contract baseline v1.0

**Status:** `APPROVED` by Tech Lead for scope control. Không có API Python nào được coi là đã triển khai. Các kết luận về hành vi upstream được đối chiếu từ [upstream audit](upstream-audit.md) và [ADR-0002](decisions/0002-m1-contract-baseline.md); những lựa chọn group interface là specification cho code nhóm, không phải khả năng đã thử nghiệm.

## Baseline đã chốt cho biên module

| Contract | Provider → consumer | Baseline v1.0 và giới hạn evidence |
| --- | --- | --- |
| C-01 Model loading | WS1 → WS3, WS4 | `ModelSpec` chỉ rõ architecture/config/checkpoint/device; strict loading, metadata/checksum, báo lỗi mismatch. Checkpoint schema thực phải được WS1 audit. |
| C-02 Image I/O | WS2 → WS1, WS3, WS4 | RGB `float32` `[0,1]`; single-image CHW `[3,H,W]`; model-facing batch NCHW `[N,3,H,W]`. Conversion explicit; reflection pad right/bottom tới bội 32 theo upstream test, crop về H×W. Trường hợp ảnh quá nhỏ cần policy/error rõ. |
| C-03 Dataset | WS2 → WS3, WS4 | Manifest ID ổn định, cặp input/GT 1:1 khi paired, có ảnh không GT; deterministic ordering; không resize/crop ngầm. Layout upstream test `test/input` và `test/target` chỉ là adapter option. |
| C-04 Inference | WS3 → WS4 | Nhận model + sample, trả restored image đúng H×W, checkpoint/device/padding/time metadata khi đo; tách việc chạy inference khỏi phần ghi ảnh; không tự claim CPU support. |
| C-05 Evaluation | WS4 ← WS2, WS3 | Metric yêu cầu aligned GT; có ID phiên bản protocol/dataset, crop/color/range/quantization. PSNR/SSIM theo từng protocol được audit; RealBlur cần kiểm code đánh giá chính thức. |
| C-06 Config | WS4 → WS1–WS3 | Schema versioned, path resolve tại runtime (Windows/Linux), checkpoint/dataset reference, metadata đủ để lặp lại thí nghiệm, không secret/path máy cá nhân trong file commit. |

**Các lựa chọn chưa được kiểm chứng:** checkpoint format/strict load thực, CPU inference, tương thích Python/PyTorch, protocol đánh giá RealBlur, policy ảnh nhỏ và thời gian GPU. Không đánh dấu `PASS` trước khi có test.

## Ownership và quyết định scope

- C-01: owner WS1, consumer reviewer WS3.
- C-02: owner WS2, consumer reviewers WS1 + WS3.
- C-03: owner WS2, consumer reviewer WS4.
- C-04: owner WS3, consumer reviewers WS2 + WS4.
- C-05: owner WS4, consumer reviewer WS2.
- C-06: owner WS4, consumer reviewers WS1 + WS2 + WS3.

`docs/ownership.md` hiện để `TBD`; Tech Lead giao implementation theo workstream sau. Consumer feedback cần được ghi trong task/PR, nhưng chỉ Tech Lead có quyền quyết định thay đổi contract hoặc mở rộng phạm vi. Không thành viên nào tự thay đổi interface chung để làm nhanh task cá nhân.

## Dependency gate

1. **Wave A:** WS1 audit architecture/checkpoint, WS2 audit dataset/preprocessing, WS3 thiết kế CLI/wrapper, WS4 audit protocol/metadata. Đây là evidence cho specification v1.0.
2. **Wave B:** **COMPLETE** — Tech Lead chốt C-01..C-06 v1.0 theo ADR-0002. Không đợi gán GitHub username để bắt đầu task đã được giao.
3. **Wave C:** code module theo contract đã chốt, thêm unit/integration tests và PR riêng theo scope Tech Lead giao; mọi thay đổi interface phải quay lại Tech Lead trước khi code.

Vòng cũ `WS3 design → contract approved → WS3 design` đã được gỡ: WS3 design độc lập ở Wave A; implementation tuân contract đã freeze và scope Tech Lead giao ở Wave C.

## Quy tắc áp dụng sau approval

- [x] Tech Lead đã chốt RGB/BGR, HWC/CHW/NCHW, dtype/range, pad/crop và xử lý ảnh không GT trong C-02/C-03/C-04.
- [x] Tech Lead đã chốt checkpoint identity, metadata và error/device policy trong C-01/C-04/C-06.
- [x] C-05 phân tách `upstream-basicsr` và `realblur-official`; không lấy PSNR/SSIM paper làm số đo nhóm.
- [x] Backlog không còn dependency vòng; Wave C chỉ khởi động theo scope Tech Lead giao.
- [ ] Mỗi task implementation phải có test plan/failure cases và chỉ ghi test thực sự chạy.
- [ ] Không sửa `third_party/FFTformer`, không commit weights/dataset.

## Trạng thái và next actions

1. Tech Lead giao task Wave C theo workstream và scope cụ thể.
2. Mỗi workstream mở PR nhỏ, có test/evidence đúng module; feedback consumer không tự thay đổi contract.
3. Nếu evidence mới mâu thuẫn v1.0, dừng thay đổi interface và xin quyết định Tech Lead.

**Out of scope M1.0:** training, thuê GPU, tải dataset/checkpoint lớn, benchmark, thay đổi cấu trúc FFTformer gốc. Điều đó không cản trở việc nghiên cứu code và thí nghiệm nhỏ độc lập khi được giao.
