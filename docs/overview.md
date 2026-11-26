# Tổng quan dự án

## Bối cảnh và mục tiêu bài toán

Ảnh có thể bị mờ do chuyển động của camera/chủ thể, mất nét hoặc các yếu tố trong quá trình thu nhận. Mục tiêu dài hạn của dự án là nghiên cứu cách FFTformer khôi phục ảnh rõ từ ảnh mờ và đánh giá việc tái hiện theo một quy trình thực nghiệm có kiểm chứng.

## Phạm vi nghiên cứu

Nhóm đã phân tích kiến trúc FFTformer, pin mã upstream và tích hợp bốn module model, data, inference và evaluation. Phạm vi kiểm chứng gồm checkpoint GoPro trên CPU và các fixture nhỏ. [Trạng thái dự án](project-status.md)

## Đầu vào và đầu ra

CLI suy luận nhận ảnh RGB, thư mục ảnh hoặc manifest và lưu ảnh đầu ra cùng metadata tùy chọn. CLI đánh giá nhận các cặp prediction/ground truth đã có sẵn và ghi PSNR/SSIM theo protocol đã chọn. [Suy luận](modules/inference.md) · [Đánh giá](modules/evaluation.md)

## Nội dung đã tìm hiểu

- Mô hình tạo ảnh mờ cơ bản `B = I * K + N`.
- Sự khác biệt giữa miền không gian và miền tần số.
- Vai trò của FFT và IFFT.
- Khái niệm Q, K, V trong Self-Attention.
- Ý tưởng cơ bản của FSAS trong công trình FFTformer.

## Tài liệu nghiên cứu

- Kiến trúc FFTformer, gồm FSAS và DFFN. [Phân tích kiến trúc](research/fftformer-architecture.md)
- Nguồn dữ liệu, layout và điều kiện đánh giá. [Audit dữ liệu](research/ws2-dataset-audit.md) · [Metric protocols](ws4/metric-protocols.md)
