# Evaluation

**Trạng thái:** tài liệu này ghi lại audit giao thức upstream. WS4 đã chạy PSNR/SSIM theo `group-basicsr-cpu` trên fixture tổng hợp. Các điểm benchmark theo giao thức upstream và RealBlur official vẫn `UNVERIFIED`. [Đánh giá của nhóm](../modules/evaluation.md)

## Điều upstream khai báo

`options/train/GoPro.yml` và `Realblur.yml` cấu hình BasicSR `calculate_psnr` và `calculate_ssim` với `crop_border: 0`, `test_y_channel: false`. BasicSR validation có thể tính metric trên output và GT. README upstream ghi PSNR RealBlur được đo bằng [official RealBlur evaluation code](https://github.com/rimchang/RealBlur#evaluation), không phải loop `test.py`.

Do hai con đường này có thể khác protocol, không được dùng metric tự tính hoặc config khác để khẳng định benchmark paper đã được tái hiện. WS4 phải audit primary evaluator và chốt protocol per benchmark.

## Preconditions cho mọi metric

1. Prediction và GT cùng sample, spatial size, color space và range theo protocol.
2. Crop border, color conversion, quantization/save-load policy được version trong config.
3. Dataset split và checkpoint identity được ghi traceable.
4. Báo cáo ghi rõ source: paper / group-measured / `UNVERIFIED`.

## Benchmark bổ sung

- **GoPro.** Dùng split test, checkpoint GoPro và cặp prediction/GT cùng tên. Ghi giao thức đã dùng. Kết quả từ CLI nhóm với `group-basicsr-cpu` được báo cáo theo giao thức của nhóm. Khi đối chiếu kết quả upstream, kiểm tra cấu hình BasicSR, cách lượng tử hóa và thiết lập SSIM trước khi so điểm. [Nguồn dữ liệu GoPro](https://seungjunnah.github.io/Datasets/gopro) · [Metric protocols](../ws4/metric-protocols.md)
- **RealBlur.** Xử lý riêng RealBlur-J và RealBlur-R, xác minh checkpoint cho từng split và chạy evaluator chính thức có căn chỉnh ECC. [RealBlur Evaluation](https://github.com/rimchang/RealBlur#evaluation)

## Mẫu ghi nhận kết quả

```text
benchmark + split:
evaluator/protocol commit or version:
prediction source + group SHA:
checkpoint + checksum/source:
color/range/crop policy:
count valid samples / exclusions:
PSNR/SSIM: <only measured values>
command + environment:
deviation from upstream/paper:
```

Không tính metric khi thiếu GT. Khi kết quả sai paper, ghi sai khác và điều tra config/data/protocol trước khi kết luận.
