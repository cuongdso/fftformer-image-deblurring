# Tài liệu tham khảo

## Công trình gốc

1. Lingshun Kong, Jiangxin Dong, Jianjun Ge, Mingqiang Li, và Jinshan Pan. “Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring.” CVPR 2023. [PDF chính thức](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf).

## Implementation chính thức

- [kkkls/FFTformer](https://github.com/kkkls/FFTformer), repository của tác giả được mô tả là official PyTorch implementation. Tại thời điểm khảo sát M0, repository này công bố [MIT License](https://github.com/kkkls/FFTformer/blob/main/license).

  - Baseline của nhóm pin tại commit [`01306b5d4072146a011c8c6e9d4927cf4607f1d5`](https://github.com/kkkls/FFTformer/commit/01306b5d4072146a011c8c6e9d4927cf4607f1d5); xem [upstream audit](upstream-audit.md).
- [RealBlur evaluation repository](https://github.com/rimchang/RealBlur#evaluation), được README upstream dẫn cho PSNR RealBlur. `UNVERIFIED` tại M0: nhóm chưa clone/chạy evaluator này.

## Phạm vi của repository nhóm

Repository này là công việc học tập độc lập của nhóm sinh viên. M0 không sao chép mã nguồn, pretrained weights, dataset hoặc kết quả thực nghiệm từ công trình/repository gốc. Việc sử dụng lại mã trong tương lai phải tuân thủ giấy phép của nguồn tương ứng và ghi nhận đầy đủ tác giả.

Các mô tả kỹ thuật trong repository này dựa chủ yếu trên paper và README của implementation chính thức nêu trên. Những nguồn bổ sung chỉ được thêm sau khi được nhóm xác minh.
