# Môi trường cho official baseline

**Trạng thái:** hướng dẫn được đối chiếu tĩnh với upstream pin `01306b5…`; **UNVERIFIED** vì M0 không cài dependency hoặc chạy baseline.

## Thông tin đã xác minh từ upstream

- README nêu PyTorch 1.11, Python, scikit-image, opencv-python, TensorBoard và einops.
- `requirements.txt` không pin version và không bao gồm `torch`/`torchvision`.
- `test.sh`/`train.sh` chạy `python setup.py develop --no_cuda_ext`; tùy chọn này không build CUDA extensions.
- Training config đặt 8 GPU và NCCL; test script hard-code `cuda`. Linux + NVIDIA GPU là giả định mạnh của code, nhưng upstream không công bố matrix OS/CUDA/driver đã test.

## Quy trình đề xuất, chưa chạy

Chỉ thực hiện trong environment riêng sau khi quyết định Python/PyTorch/CUDA tương thích và đọc rủi ro tại [upstream audit](../upstream-audit.md):

```bash
git submodule update --init --recursive
cd third_party/FFTformer
python -m venv .venv
# Linux/macOS shell: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Cài PyTorch/torchvision theo bảng chính thức của PyTorch cho CUDA/driver thực tế.
python -m pip install -r requirements.txt
python setup.py develop --no_cuda_ext
python ../../scripts/check_env.py --require-torch
```

Lệnh trên là hướng dẫn audit-derived, không phải log cài đặt thành công. `setup.py` import `torch` trước khi chạy, do đó cài PyTorch là precondition. Không cài/biên dịch extension CUDA trừ khi task xác minh toolchain cần nó.

## Xác minh môi trường trước khi chạy

Từ root nhóm, chạy `python scripts/check_env.py`; script chỉ report fact, không cài gì. Trước test upstream, lưu Python/PyTorch/CUDA/driver/GPU vào metadata thí nghiệm. `CUDA available: True` không chứng minh checkpoint/dataset/protocol hợp lệ.

## Compatibility và xử lý xung đột

- `UNVERIFIED`: Python version chính xác; hãy ghi traceback và environment facts vào PR/handoff hoặc trao đổi nhóm thay vì đổi version tùy ý.
- Với lỗi build extension, giữ `--no_cuda_ext` như script upstream trước khi thử thay đổi build.
- Với CUDA unavailable, `test.py` upstream không có CPU fallback; ghi blocker, không patch baseline để gọi đó là reproduction.
- Không khẳng định Windows, CPU-only hoặc PyTorch hiện đại được hỗ trợ cho đến khi có evidence.
