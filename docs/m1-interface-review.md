# M1.0 — Interface alignment / review candidate v0.1

**Status:** `PROPOSED` — cần review chéo trước khi chuyển từng contract trong [contracts.md](contracts.md) sang `APPROVED`. Không có API Python nào được coi là đã triển khai. Các kết luận về hành vi upstream được đối chiếu từ [upstream audit](upstream-audit.md); những lựa chọn group interface dưới đây là **đề xuất thiết kế**, không phải khả năng đã thử nghiệm.

## Quyết định đề xuất cho biên module

| Contract | Provider → consumer | Candidate v0.1 và điểm cần xác nhận |
| --- | --- | --- |
| C-01 Model loading | WS1 → WS3, WS4 | `ModelSpec` chỉ rõ architecture/config/checkpoint/device; strict loading, metadata/checksum, báo lỗi mismatch. Checkpoint schema thực phải được WS1 audit. |
| C-02 Image I/O | WS2 → WS1, WS3, WS4 | RGB `float32` `[0,1]`; single-image CHW `[3,H,W]`; model-facing batch NCHW `[N,3,H,W]`. Conversion explicit; reflection pad right/bottom tới bội 32 theo upstream test, crop về H×W. Trường hợp ảnh quá nhỏ cần policy/error rõ. |
| C-03 Dataset | WS2 → WS3, WS4 | Manifest ID ổn định, cặp input/GT 1:1 khi paired, có ảnh không GT; deterministic ordering; không resize/crop ngầm. Layout upstream test `test/input` và `test/target` chỉ là adapter option. |
| C-04 Inference | WS3 → WS4 | Nhận model + sample, trả restored image đúng H×W, checkpoint/device/padding/time metadata khi đo; tách việc chạy inference khỏi phần ghi ảnh; không tự claim CPU support. |
| C-05 Evaluation | WS4 ← WS2, WS3 | Metric yêu cầu aligned GT; có ID phiên bản protocol/dataset, crop/color/range/quantization. PSNR/SSIM theo từng protocol được audit; RealBlur cần kiểm code đánh giá chính thức. |
| C-06 Config | WS4 → WS1–WS3 | Schema versioned, path resolve tại runtime (Windows/Linux), checkpoint/dataset reference, metadata đủ để lặp lại thí nghiệm, không secret/path máy cá nhân trong file commit. |

**Các lựa chọn chưa được kiểm chứng:** checkpoint format/strict load thực, CPU inference, tương thích Python/PyTorch, protocol đánh giá RealBlur, policy ảnh nhỏ và thời gian GPU. Không đánh dấu `PASS` trước khi có test.

## Ai duyệt gì?

- C-01: owner WS1, consumer reviewer WS3.
- C-02: owner WS2, consumer reviewers WS1 + WS3.
- C-03: owner WS2, consumer reviewer WS4.
- C-04: owner WS3, consumer reviewers WS2 + WS4.
- C-05: owner WS4, consumer reviewer WS2.
- C-06: owner WS4, consumer reviewers WS1 + WS2 + WS3.

`docs/ownership.md` hiện để `TBD`; cần gán username thật. Một thành viên không được tự approve contract của chính mình thay người consumer. Quyết định ghi trên issue/PR và liên kết tại đây.

## Dependency gate — không block nghiên cứu độc lập

1. **Wave A:** WS1 audit architecture/checkpoint, WS2 audit dataset/preprocessing, WS3 thiết kế CLI/wrapper, WS4 audit protocol/metadata. Đọc `PROPOSED` như working draft; không code shared interface trước approval.
2. **Wave B:** họp/review chéo Wave A, giải quyết bất đồng shape/range/device/metadata, cập nhật C-01..C-06 rồi gắn reviewer evidence để chuyển từng contract thành `APPROVED`.
3. **Wave C:** code module theo contract đã chốt, thêm unit/integration tests và PR riêng cho mỗi workstream; khi thay interface phải quay lại review.

Vòng cũ `WS3 design → contract approved → WS3 design` đã được gỡ: WS3 design độc lập ở Wave A, **implementation phụ thuộc approved** ở Wave C.

## Acceptance cho PR chốt contract

- [ ] Có owner/reviewer thật ở `docs/ownership.md`.
- [ ] Từng contract được owner và consumer reviewer xác nhận trên PR/issue.
- [ ] Chốt rõ RGB/BGR, HWC/CHW/NCHW, dtype/range, pad/crop, xử lý ảnh không GT.
- [ ] Thống nhất checkpoint identity, metadata, error/device policy.
- [ ] Mỗi contract có ít nhất test plan và failure cases; test đã chạy ghi kết quả thật, không tự đánh dấu PASS.
- [ ] C-05 nêu source/version metric protocol, không lấy PSNR/SSIM paper làm số đo nhóm.
- [ ] M1 backlog không còn dependency vòng và không có task phát triển shared API trước approval.
- [ ] Không sửa `third_party/FFTformer`, không commit weights/dataset.

## Trạng thái và next actions

1. Gán 4 GitHub usernames tương ứng WS1–WS4, mỗi contract có người review độc lập.
2. Chuyển các đề xuất Wave A thành issue riêng sau khi gán owner.
3. Review candidate v0.1, giải quyết điểm `UNVERIFIED` trước khi chốt API và mở PR implementation.

**Out of scope M1.0:** training, thuê GPU, tải dataset/checkpoint lớn, benchmark, thay đổi cấu trúc FFTformer gốc. Điều đó không cản trở việc nghiên cứu code và thí nghiệm nhỏ độc lập khi được giao.
