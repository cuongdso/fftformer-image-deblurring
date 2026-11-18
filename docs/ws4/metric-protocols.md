# Giao thức đo PSNR và SSIM

## Vai trò trong image deblurring

PSNR và SSIM so sánh ảnh khôi phục với ground truth đã được ghép đúng mẫu và căn chỉnh theo protocol. Chúng không tự chứng minh mô hình tốt nếu sai dataset split, checkpoint, color conversion hoặc alignment. WS4-02 vì thế gắn mỗi con số với `MetricProtocol` và chỉ ghi nhãn `group-measured`.

## PSNR

Với hai ảnh có sai số bình phương trung bình

\[
\operatorname{MSE}=\frac{1}{N}\sum_{i=1}^{N}(x_i-y_i)^2,
\]

PSNR là

\[
\operatorname{PSNR}=10\log_{10}\left(\frac{MAX^2}{\operatorname{MSE}}\right)
=20\log_{10}\left(\frac{MAX}{\sqrt{\operatorname{MSE}}}\right).
\]

Trong representation HWC `uint8`, `MAX=255` và đơn vị là dB. Khi MSE bằng 0, PSNR là `+inf`; WS4-02 coi đó là kết quả toán học hợp lệ và serialize rõ ràng thay vì báo lỗi.

## SSIM

SSIM đánh giá mức tương đồng cấu trúc thay vì chỉ sai số pixel. Công thức cục bộ kết hợp ba yếu tố: luminance so sánh trung bình, contrast so sánh phương sai, và structure so sánh hiệp phương sai. BasicSR sử dụng cửa sổ Gaussian 11×11 với sigma 1.5; các hằng số ổn định là \(C_1=(0.01MAX)^2\), \(C_2=(0.03MAX)^2\). Cửa sổ, padding và cách average đều ảnh hưởng kết quả.

## Representation BasicSR đã xác minh

Tại pin FFTformer, validation dùng `tensor2img` trước evaluator. Tensor RGB CHW/NCHW được clamp vào `[0,1]`, đổi sang HWC, đổi RGB sang BGR khi `rgb2bgr=True`, sau đó nhân 255 và làm tròn thành `uint8`. `GoPro.yml` và `Realblur.yml` khai báo `crop_border: 0`, `test_y_channel: false`.

WS4-02 áp dụng representation này cho cả prediction lẫn GT. Điều đó giải thích vì sao engine không gọi metric với tensor NCHW. Clamp và quantization có thể thay đổi giá trị biên và do đó thay đổi PSNR/SSIM; crop border loại pixel viền khỏi phép đo nên cũng phải là phần của protocol, không chỉ là tham số tiện dụng.

## Ba protocol không thay thế nhau

| Protocol | Điều đã xác minh từ mã nguồn | Phạm vi sử dụng |
| --- | --- | --- |
| `upstream-basicsr` | `calculate_psnr` và `calculate_ssim` upstream; SSIM mặc định `ssim3d=True` gọi CUDA | Chỉ chạy khi caller cung cấp adapter BasicSR/CUDA tường minh. |
| `group-basicsr-cpu` | Cùng conversion/PSNR, SSIM dùng công thức Gaussian 2D tương ứng `ssim3d=False` | Đo CPU của nhóm; không phải kết quả upstream mặc định. |
| `realblur-official` | Evaluator chính thức dùng intensity matching, ECC homography, mask và SSIM/PSNR riêng | Đăng ký nhưng chưa khả dụng ở WS4-02. |

BasicSR CPU variant được tạo vì default SSIM upstream không CPU-safe. Sự thay đổi `ssim3d=True` sang `False` là thay đổi thuật toán, nên protocol ID khác và engine cấm fallback ngầm.

RealBlur official không chỉ là PSNR/SSIM trên cặp ảnh có sẵn: source chính thức căn chỉnh prediction với GT bằng ECC, tính mask hợp lệ và dùng dependency legacy. Vì vậy BasicSR không thể dùng thay thế để tuyên bố kết quả RealBlur official.

## Ground truth, alignment và khả năng so sánh

Engine yêu cầu `Sample.target` tồn tại, sample ID trùng, cùng spatial shape, RGB float32 CHW hữu hạn trong `[0,1]`. Điều này chỉ xác minh alignment hình học theo contract nhóm; nó không thay thế ECC alignment của RealBlur. Sample thiếu GT được lưu vào exclusions thay vì tham gia trung bình.

Không so sánh trực tiếp PSNR/SSIM giữa protocol khác nhau. Khác color order, quantization, crop, Y-channel, SSIM window, ECC/mask hoặc dataset split đều có thể làm con số khác dù ảnh dự đoán giống nhau.

## Mức độ tương thích và phần chưa xác minh

Đã xác minh tĩnh từ `third_party/FFTformer/basicsr/metrics/psnr_ssim.py`, `basicsr/utils/img_util.py` và YAML validation: conversion BGR/HWC/uint8, crop 0, không Y-channel, và CUDA call của SSIM 3D. `group-basicsr-cpu` được kiểm thử bằng PSNR reference đóng và fixture SSIM deterministic.

Chưa xác minh trong repository này: chạy `upstream-basicsr` SSIM CUDA với artifact thật, evaluator RealBlur official end-to-end, benchmark GoPro/RealBlur và mức tương đương với số liệu paper. Những điều này không được suy ra từ unit test WS4-02.
