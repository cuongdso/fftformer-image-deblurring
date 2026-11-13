# Troubleshooting có bằng chứng

Các mục dưới đây xuất phát từ static audit upstream; chúng không phải danh sách “cách sửa” đã được chạy ở M0.

| Symptom | Evidence | Hành động an toàn |
| --- | --- | --- |
| `CUDA` unavailable / runtime error | `test.py` tạo device CUDA và gọi CUDA APIs không điều kiện | ghi `scripts/check_env.py`, PyTorch/CUDA/driver facts; không patch baseline rồi gọi đó là baseline reproduction |
| `FileNotFoundError` dataset | loader dùng `<data_dir>/test/input` và luôn mở `target/<same-name>` | kiểm tra paired layout, path và filename; không bỏ target trong baseline test |
| `load_state_dict` mismatch | `strict=True` trong `test.py`; default test model filename không quan sát trong pretrain dir audit | ghi exact checkpoint/source/SHA; đối chiếu config/model trước khi đổi strictness |
| Import/build error | `setup.py` import torch nhưng `requirements` không cài torch; install_requires rỗng | cài PyTorch/torchvision phù hợp trước, report versions; giữ `--no_cuda_ext` như official script khi audit |
| Shape/rearrange/padding error | DFFN/FSAS patch 8×8; test pad bội số 32 | lưu input H×W, giữ policy upstream trong baseline; WS3/WS1 thêm test trước wrapper |
| Out of memory | transformer/FFT và config train 8 GPU | giảm scope chỉ trong experiment riêng có metadata; không đổi baseline architecture im lặng |

Khi báo lỗi trong nhóm hoặc PR, đính kèm command, commit SHA, traceback đã lọc secrets, Python/PyTorch/CUDA/GPU, input layout/shape và checkpoint/dataset identity.
