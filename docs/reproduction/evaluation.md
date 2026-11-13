# Evaluation

**Trạng thái:** không có metric group nào đã chạy ở M0; không có PSNR/SSIM của nhóm.

## Điều upstream khai báo

`options/train/GoPro.yml` và `Realblur.yml` cấu hình BasicSR `calculate_psnr` và `calculate_ssim` với `crop_border: 0`, `test_y_channel: false`. BasicSR validation có thể tính metric trên output và GT. README upstream ghi PSNR RealBlur được đo bằng [official RealBlur evaluation code](https://github.com/rimchang/RealBlur#evaluation), không phải loop `test.py`.

Do hai con đường này có thể khác protocol, không được dùng metric tự tính hoặc config khác để khẳng định benchmark paper đã được tái hiện. WS4 phải audit primary evaluator và chốt protocol per benchmark.

## Preconditions cho mọi metric

1. Prediction và GT cùng sample, spatial size, color space và range theo protocol.
2. Crop border, color conversion, quantization/save-load policy được version trong config.
3. Dataset split và checkpoint identity được ghi traceable.
4. Báo cáo ghi rõ source: paper / group-measured / `UNVERIFIED`.

## Template ghi nhận kết quả (PLANNED)

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
