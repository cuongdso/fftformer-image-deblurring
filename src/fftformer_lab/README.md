# Group implementation

- [`model/`](model/README.md) nạp kiến trúc từ submodule và strict-load checkpoint.
- [`data/`](data/README.md) đọc ảnh RGB và cung cấp dataset theo manifest.
- [`inference/`](inference/README.md) chạy pipeline khử mờ và lưu ảnh bằng CLI.
- [`evaluation/`](evaluation/README.md) tính PSNR/SSIM và ghi metadata.

Interface liên module được mô tả trong [contracts](../../docs/contracts.md).

- `model/`: WS1
- `data/`: WS2
- `inference/`, `app/`: WS3
- `evaluation/`: WS4
- `common/`: only shared utilities approved through an interface discussion
