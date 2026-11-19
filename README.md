# FFTformer Image Deblurring

Mã nguồn bài tập lớn môn **Xử lý ảnh số** về khử mờ ảnh với FFTformer. Repository gồm mã pipeline do nhóm xây dựng và bản mã nguồn chính thức FFTformer để chạy đối chiếu. Không kèm dataset, checkpoint hay kết quả thực nghiệm.

## Cấu trúc

```text
src/fftformer_lab/       Pipeline của nhóm: đọc ảnh, model adapter, inference, evaluation
third_party/FFTformer/   Mã nguồn FFTformer gốc (đã đưa trực tiếp vào repository)
configs/                 Schema và cấu hình evaluation mẫu
scripts/                 Kiểm tra môi trường và script xác minh nhẹ
tests/                   Unit test và fixture ảnh nhỏ
```

## Môi trường và cài đặt

Cần Python 3.10 trở lên. Tạo môi trường ảo riêng, rồi cài package của nhóm và dependency xử lý ảnh:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data]"
```

Mã upstream có dependency riêng tại `third_party/FFTformer/requirements.txt`. Khi cần chạy trực tiếp entrypoint upstream, cài chúng theo hướng dẫn trong `third_party/FFTformer/README.md`. Phiên bản PyTorch/CUDA phải phù hợp với máy và checkpoint đang dùng.

## Inference

Lệnh dưới chạy pipeline nhóm với mock model để kiểm tra luồng I/O; đây **không** phải kết quả FFTformer thật:

```bash
python -m fftformer_lab.inference --input path/to/blur.png --output outputs/restored.png --mock
```

Để nạp checkpoint FFTformer, chuẩn bị file checkpoint hợp lệ bên ngoài repository và chỉ rõ đường dẫn của nó. CLI strict-load checkpoint và mặc định dùng architecture upstream trong thư mục `third_party/FFTformer/`:

```bash
python -m fftformer_lab.inference --input path/to/blur.png --output outputs/restored.png --checkpoint path/to/fftformer_checkpoint.pth --device cuda
```

Checkpoint và dataset không được commit. Hãy tự tải từ nguồn phát hành hợp lệ, kiểm tra điều khoản sử dụng, và lưu ngoài thư mục repository hoặc trong các đường dẫn đã ignore. Máy không có CUDA có thể dùng `--device cpu` cho pipeline nhóm; khả năng tương thích thực tế phụ thuộc checkpoint và môi trường PyTorch.

## Evaluation

Evaluation nhận ảnh dự đoán và ground truth đã có sẵn theo file cấu hình JSON:

```bash
python -m fftformer_lab.evaluation.cli validate-config configs/examples/evaluation-precomputed.json
python -m fftformer_lab.evaluation.cli run path/to/evaluation.json
```

Xem `configs/evaluation.schema.json` để biết các trường bắt buộc. Kết quả metric chỉ có ý nghĩa theo protocol và dữ liệu bạn cung cấp; repository này không kèm benchmark GoPro hoặc RealBlur.

## Kiểm tra nhanh

```bash
python scripts/check_env.py
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q scripts src third_party/FFTformer
```

## Ghi công FFTformer

FFTformer gốc: Lingshun Kong và cộng sự, *Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring*, CVPR 2023. Mã nguồn upstream: [kkkls/FFTformer](https://github.com/kkkls/FFTformer), giấy phép MIT. Khi sử dụng hoặc phát hành lại, giữ nguyên thông báo bản quyền và giấy phép trong `third_party/FFTformer/license`.
