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

## Chạy thật với một ảnh bị mờ (Windows PowerShell)

Đây là cách chạy chương trình để tạo **một ảnh đã khử mờ**, không phải test và không dùng mock model. Checkpoint của nhóm được lưu trên [Google Drive](https://drive.google.com/file/d/1bFuh1IgbZL-YqtBQyG--0cbJXyTlx__V/view?usp=sharing), tách khỏi source code do dung lượng lớn. Tải file đó về và lưu tại `checkpoints\fftformer_checkpoint.pth` trong thư mục project.

Mở PowerShell tại thư mục repository, rồi chạy lần lượt:

```powershell
# Chỉ cần làm một lần khi tạo môi trường.
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data]"
python -m pip install einops
```

Đặt ảnh bị mờ bất kỳ (PNG/JPG RGB) vào `input\blur.png`, tải checkpoint theo link phía trên, rồi chạy trên CPU:

```powershell
New-Item -ItemType Directory -Force .\input, .\outputs, .\checkpoints | Out-Null
python -m fftformer_lab.inference `
  --input ".\input\blur.png" `
  --output ".\outputs\restored.png" `
  --checkpoint ".\checkpoints\fftformer_checkpoint.pth" `
  --device cpu
```

Ảnh đã khử mờ nằm tại `outputs\restored.png`. Nếu máy có PyTorch CUDA hoạt động, đổi `--device cpu` thành `--device cuda` để dùng GPU. Không thêm `--mock`: tùy chọn đó chỉ kiểm tra luồng I/O và **không** chạy FFTformer thật.

Nếu báo lỗi khi nạp checkpoint, kiểm tra lại file đã tải xong và đúng đường dẫn `checkpoints\fftformer_checkpoint.pth`. CLI cố ý strict-load nên không chạy tiếp với trọng số sai. Checkpoint và dataset không được commit vào Git; bản nộp Google Drive có thể để checkpoint cạnh source code theo cấu trúc trên.

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
