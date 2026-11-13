# Tổng quan dự án

## Bối cảnh và mục tiêu bài toán

Ảnh có thể bị mờ do chuyển động của camera/chủ thể, mất nét hoặc các yếu tố trong quá trình thu nhận. Mục tiêu dài hạn của dự án là nghiên cứu cách FFTformer khôi phục ảnh rõ từ ảnh mờ và đánh giá việc tái hiện theo một quy trình thực nghiệm có kiểm chứng.

## Phạm vi nghiên cứu

Trong giai đoạn M0, phạm vi chỉ gồm đọc tài liệu, xây dựng ghi chú và chuẩn bị repository. Chưa có mã mô hình, dataset, weights, huấn luyện, đánh giá benchmark hay ứng dụng minh họa.

## Đầu vào và đầu ra dự kiến

Ở các giai đoạn sau, đầu vào dự kiến là ảnh bị mờ; đầu ra dự kiến là ảnh đã được khôi phục độ rõ bởi một quy trình được ghi chép. Chất lượng đầu ra và điều kiện dữ liệu chưa được khẳng định ở M0.

## Nội dung đã tìm hiểu

- Mô hình tạo ảnh mờ cơ bản `B = I * K + N`.
- Sự khác biệt giữa miền không gian và miền tần số.
- Vai trò của FFT và IFFT.
- Khái niệm Q, K, V trong Self-Attention.
- Ý tưởng cơ bản của FSAS trong công trình FFTformer.

## Nội dung cần tiếp tục nghiên cứu

- Kiến trúc FFTformer chi tiết, vị trí và luồng dữ liệu của các khối.
- Thiết kế, giả định và triển khai DFFN.
- Thiết lập dữ liệu, giao thức đánh giá và cách tái hiện có trách nhiệm.
- Các giới hạn, chi phí tính toán và điều kiện áp dụng của phương pháp.
