# Ghi chú paper FFTformer

Nguồn chuẩn: [paper CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf). Đây là ghi chú nhập môn, không phải xác nhận implementation/benchmark nhóm.

- Mô hình ảnh mờ thường ghi `B = I * K + N`.
- FFT/IFFT chuyển giữa miền không gian và tần số; paper dùng cảm hứng từ định lý chập.
- Self-Attention dựa Q/K/V; paper đề xuất FSAS với thao tác frequency-domain.
- DFFN dùng gating và xử lý frequency-domain; chi tiết cần tiếp tục đối chiếu paper/code.

Xem `fftformer-architecture.md` cho mapping code đã audit và `upstream-audit.md` cho giới hạn evidence.
