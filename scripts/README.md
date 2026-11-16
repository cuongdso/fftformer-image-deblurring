# Tooling M0

- `check_env.py`: kiểm tra thông tin Python, OS, package tùy chọn và CUDA nếu PyTorch đã cài. Không cài dependency hay tải artifact.
- `bootstrap.sh`: hướng dẫn POSIX tối thiểu tạo virtual environment và chạy checker. `UNVERIFIED` trên mọi OS ở M0.

Các script không thay thế quy trình cài đặt đã audit của upstream tại `docs/reproduction/environment.md`.

WS1 bổ sung các script opt-in cần PyTorch/einops và upstream source/checkpoint sẵn có:

- `ws1_01_verify.py`: kiểm tra kiến trúc/checkpoint trực tiếp trên upstream.
- `ws1_03_verify.py`: chạy bộ test WS1-02/03, gồm fixture mock và GoPro integration,
  có thể xuất JSON evidence theo [model guide](../docs/modules/model.md#kiểm-thử-ws1-03-và-evidence-hiện-tại).

Các script này không tải artifact, training hoặc benchmark chất lượng và không chạy trong CI mặc định.
