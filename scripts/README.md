# Tooling M0

- `check_env.py`: kiểm tra thông tin Python, OS, package tùy chọn và CUDA nếu PyTorch đã cài. Không cài dependency hay tải artifact.
- `bootstrap.sh`: hướng dẫn POSIX tối thiểu tạo virtual environment và chạy checker. `UNVERIFIED` trên mọi OS ở M0.

Các script không thay thế quy trình cài đặt đã audit của upstream tại `docs/reproduction/environment.md`.
