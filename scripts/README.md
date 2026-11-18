# Tooling M0

- `check_env.py`: kiểm tra thông tin Python, OS, package tùy chọn và CUDA nếu PyTorch đã cài. Không cài dependency hay tải artifact.
- `bootstrap.sh`: hướng dẫn POSIX tối thiểu tạo virtual environment và chạy checker. `UNVERIFIED` trên mọi OS ở M0.

Các script không thay thế quy trình cài đặt đã audit của upstream tại `docs/reproduction/environment.md`.

WS1 bổ sung các script opt-in cần PyTorch/einops và upstream source/checkpoint sẵn có:

- `ws1_01_verify.py`: kiểm tra kiến trúc/checkpoint trực tiếp trên upstream.
- `ws1_03_verify.py`: chạy bộ test WS1-02/03, gồm fixture mock và GoPro integration,
  có thể xuất JSON evidence theo [model guide](../docs/modules/model.md#kiểm-thử-ws1-03-và-evidence-hiện-tại).

## WS3 Tooling
WS3 bổ sung script kiểm chứng pipeline suy luận và thu thập bằng chứng runtime evidence:

- `ws3_verify.py`: chạy toàn bộ bộ test WS3 (pure unit tests, mock pipeline và contract enforcement),
  tự động tính SHA-256 các file mã nguồn và xuất JSON evidence tại `docs/research/ws3-02-runtime-evidence.json` theo [inference guide](../docs/modules/inference.md#kiểm-thử-ws3-02-và-evidence-hiện-tại).

Các script này không tải artifact, training hoặc benchmark chất lượng và không chạy trong CI mặc định.

## WS2 tooling

`ws2_verify.py` tái lập WS2 tests, toàn suite, standard-library suite, compile,
dependency/whitespace checks và ghi stdout/stderr/exit codes cùng source hashes.
`--checkpoint <local-path>` bật model tests và smoke synthetic CPU qua WS1/WS3.
`--examples-dir <local-root>` kiểm hai pair Flower/Istanbul theo byte count và
Git blob ID từ repo tác giả GoPro đã pin.

Mặc định không tải mạng. Chỉ `--download-examples` (kèm `--examples-dir`) mới tải
bốn PNG nhỏ tổng khoảng 5 MB; không ghi đè file khác checksum, lưu attribution,
không tải full dataset/weights và không đo benchmark. Dữ liệu đặt dưới ignored
`data/`. [Handoff WS2](../docs/research/ws2-handoff.md) có lệnh đầy đủ.
