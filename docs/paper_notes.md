# Ghi chú đọc paper FFTformer

Nguồn chính: Kong et al., *Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring*, CVPR 2023. Các ghi chú này ở mức nhập môn và sẽ được cập nhật khi nhóm kiểm chứng thêm từ paper và implementation chính thức.

## Đã tìm hiểu ở mức nhập môn

### Mô hình ảnh mờ

`B = I * K + N`: ảnh mờ `B` được mô hình hóa từ ảnh rõ tiềm ẩn `I`, kernel làm mờ `K` và nhiễu `N`. Mô hình là một cách mô tả hữu ích, không bao quát toàn bộ tình huống thực tế.

### Miền không gian và miền tần số

Miền không gian biểu diễn giá trị theo vị trí điểm ảnh; miền tần số biểu diễn các thành phần thay đổi chậm/nhanh của tín hiệu. Theo định lý chập, phép chập hoặc tương quan trong miền không gian có liên hệ với tích theo phần tử ở miền tần số.

### FFT và IFFT

Fast Fourier Transform (FFT) chuyển biểu diễn sang miền tần số một cách hiệu quả. Inverse FFT (IFFT) chuyển kết quả về miền ban đầu. Cần tiếp tục nghiên cứu cách xử lý tensor phức, chuẩn hóa và các quyết định triển khai liên quan.

### Q, K, V trong Self-Attention

Self-Attention tạo Query (Q), Key (K) và Value (V) từ đặc trưng đầu vào; mức độ liên hệ giữa Q và K được dùng để tổng hợp V. Cách suy diễn chi tiết theo tensor và chi phí tính toán sẽ được kiểm chứng thêm.

### Ý tưởng cơ bản của FSAS

Paper đề xuất Frequency-domain Self-Attention Solver (FSAS), được thúc đẩy bởi định lý chập, để ước lượng scaled dot-product attention bằng tích theo phần tử trong miền tần số thay cho nhân ma trận theo cách tiếp cận miền không gian. Paper đặt FSAS ở decoder của kiến trúc encoder–decoder bất đối xứng.

## Cần nghiên cứu thêm

- Chi tiết kiến trúc và các thành phần FFTformer.
- DFFN, cơ chế gating và liên hệ được paper nêu với JPEG compression.
- Quy trình train/test, lựa chọn dataset và giao thức đánh giá.
- Cách đối chiếu implementation chính thức mà không sao chép mã ngoài phạm vi cho phép của giấy phép.

Không có mục nào ở trên được đánh dấu là đã nắm vững, đã triển khai hoặc đã tái hiện thực nghiệm.
