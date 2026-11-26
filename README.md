# FFTformer Image Deblurring

Repository học thuật của nhóm cho bài tập lớn môn Xử lý ảnh, tập trung vào việc tìm hiểu và thực nghiệm phương pháp **FFTformer** cho bài toán khử mờ ảnh (*image deblurring*).

> **Trạng thái trên `main`.** Pipeline M1 của nhóm đã tích hợp WS1–WS4: nạp checkpoint, đọc ảnh và manifest, CLI khử mờ, đánh giá PSNR/SSIM và ghi metadata. Hồ sơ hiện có ghi nhận strict load và forward checkpoint GoPro trên CPU, cùng kiểm thử tích hợp bằng `MockModel` trên ảnh tổng hợp. Benchmark đầy đủ GoPro/RealBlur và CUDA forward vẫn `UNVERIFIED`. [Trạng thái và bằng chứng](docs/project-status.md)

## Bài toán Image Deblurring

Image deblurring hướng đến việc ước lượng ảnh rõ tiềm ẩn từ ảnh quan sát bị mờ. Một mô hình tạo ảnh mờ thường được viết là `B = I * K + N`, trong đó `B` là ảnh mờ, `I` là ảnh rõ tiềm ẩn, `K` là kernel làm mờ, `N` là nhiễu và `*` biểu thị phép chập. Đây là bài toán nghịch đảo khó do có thể mất thông tin và do kernel/nhiễu không luôn được biết trước.

## Tổng quan FFTformer

FFTformer (*Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring*) là công trình của Lingshun Kong và cộng sự tại CVPR 2023. Công trình khai thác định lý chập để xử lý một số phép toán Transformer trong miền tần số, với kiến trúc encoder–decoder bất đối xứng: Frequency-domain Self-Attention Solver (FSAS) được dùng ở decoder. Bài báo cũng đề xuất Discriminative Frequency-domain Feed-Forward Network (DFFN).

### Các khái niệm chính

- **FFT / IFFT:** biến đổi nhanh từ miền không gian sang miền tần số và phép biến đổi ngược để tái tạo tín hiệu.
- **Self-Attention:** cơ chế sử dụng các biểu diễn Query (Q), Key (K), Value (V) để mô hình hóa quan hệ giữa các đặc trưng.
- **FSAS:** ý tưởng ước lượng scaled dot-product attention bằng tích theo phần tử trong miền tần số, thay cho phép nhân ma trận trực tiếp trong miền không gian.
- **DFFN:** khối FFN dựa trên miền tần số với cơ chế gating, nhằm chọn lọc thông tin tần số thấp/cao phù hợp cho khôi phục ảnh rõ.

Các mô tả trên là ghi chú nhập môn dựa trên công trình gốc, không phải tuyên bố về mức độ triển khai hay tái hiện kết quả của nhóm.

## Mục tiêu nghiên cứu và thực nghiệm

- Hiểu bài toán khử mờ ảnh, biểu diễn miền không gian và miền tần số.
- Đọc, phân tích và ghi chú các thành phần của FFTformer.
- Xây dựng pipeline khử mờ và đánh giá có thể truy vết theo checkpoint, dữ liệu và protocol.

## Baseline và phạm vi mã nguồn

`third_party/FFTformer/` là Git submodule trỏ đến official implementation tại commit `01306b5d4072146a011c8c6e9d4927cf4607f1d5`. Baseline này chỉ dùng để đối chiếu và tái hiện; không sửa trực tiếp trong các task của nhóm. Mã nhóm được phát triển độc lập trong `src/fftformer_lab/`. Chiến lược và ràng buộc giấy phép được ghi tại [ADR-0001](docs/decisions/0001-upstream-baseline.md).

Khởi tạo đầy đủ checkout:

```bash
git clone --recurse-submodules https://github.com/cuongdso/fftformer-image-deblurring.git
# Nếu đã clone mà thiếu submodule:
git submodule update --init --recursive
```

## Cấu trúc repository

```text
.
├── README.md
├── AGENTS.md
├── INSTRUCTION.md             # nhận workstream/task và quy tắc coding agent
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
├── .editorconfig
├── .github/                   # PR template và CI nhẹ
├── third_party/FFTformer/     # official baseline (Git submodule, pinned)
├── src/fftformer_lab/         # group implementation WS1–WS4
├── configs/                   # schema và ví dụ config evaluation
├── scripts/                   # bootstrap và kiểm tra môi trường
├── docs/
│   ├── reproduction/          # tutorial official baseline
│   ├── modules/               # ranh giới của 4 workstream
│   ├── research/              # ghi chú paper/kiến trúc
│   └── decisions/             # Architecture Decision Records
├── data/                      # chỉ README, dữ liệu bị ignore
├── checkpoints/               # chỉ README, weights bị ignore
├── results/                   # chỉ README, kết quả bị ignore
├── experiments/               # metadata thí nghiệm, không lưu output lớn
└── tests/                     # unit/smoke test không GPU
```

## Nguồn chính

- Paper: [Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf)
- Official implementation: [kkkls/FFTformer](https://github.com/kkkls/FFTformer)

## Ghi nhận tác giả và nguồn nghiên cứu

FFTformer là công trình gốc của Lingshun Kong, Jiangxin Dong, Jianjun Ge, Mingqiang Li và Jinshan Pan. Repository này là không gian học tập của nhóm sinh viên, độc lập với tác giả; không phải implementation chính thức. Official source chỉ hiện diện như Git submodule được pin, không được sao chép vào mã nhóm `src/`. Tham khảo chi tiết và ghi chú về giấy phép có tại [docs/references.md](docs/references.md).

## Chạy khử mờ một ảnh

Từ thư mục gốc của repository trên Windows PowerShell, tạo môi trường ảo với Python 3.10 trở lên. Chuẩn bị một ảnh PNG/JPEG RGB. Ví dụ dùng checkpoint GoPro đi kèm submodule tại commit đã pin. [Hướng dẫn về checkpoint](docs/reproduction/pretrained-models.md)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[data]" einops
.\.venv\Scripts\python.exe -m fftformer_lab.inference `
    --input data/blur.png `
    --output results/restored.png `
    --checkpoint third_party/FFTformer/pretrain_model/fftformer_GoPro.pth `
    --device cpu `
    --save-metadata
```

Thay `data/blur.png` bằng đường dẫn ảnh của bạn. CLI mặc định đọc kiến trúc từ submodule đã khởi tạo. Nếu dùng checkpoint đã lưu riêng, thay đường dẫn `--checkpoint` tương ứng. Kết quả gồm `results/restored.png` và `results/restored_metadata.json`. Khi chạy với `--mock` và bỏ đối số `--checkpoint`, CLI dùng mô hình giả lập để kiểm tra luồng đọc/ghi ảnh. Chế độ thư mục và manifest được mô tả trong [hướng dẫn suy luận](docs/modules/inference.md).

## Đánh giá ảnh đã khôi phục

```powershell
.\.venv\Scripts\python.exe -m fftformer_lab.evaluation.cli validate-config configs/examples/evaluation-precomputed.json
.\.venv\Scripts\python.exe -m pip install opencv-python
```

File ví dụ dùng để kiểm tra cấu hình. Sao chép thành `configs/examples/evaluation-local.json`, rồi chỉnh `dataset.prediction_root`, `dataset.target_root` và `output_path`. Hai thư mục đầu vào phải chứa các cặp ảnh RGB cùng tên và kích thước. Đường dẫn tương đối được tính từ thư mục chứa config. [Cấu hình mẫu](configs/examples/README.md) · [Giao thức đánh giá](docs/modules/evaluation.md)

```powershell
.\.venv\Scripts\python.exe -m fftformer_lab.evaluation.cli run configs/examples/evaluation-local.json
```

## Kiểm tra và tài liệu

```bash
python -m pip install --no-deps -e .
python scripts/check_env.py
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q scripts src
git diff --check
```

Bộ kiểm tra nhẹ bỏ qua các ca thiếu dependency hoặc cần bật chế độ model/checkpoint. [Hướng dẫn kiểm thử](tests/README.md)

- Luồng dữ liệu và ranh giới module được mô tả trong [kiến trúc](docs/architecture.md) và [contracts C-01 đến C-06](docs/contracts.md).
- Quy trình đóng góp và bàn giao ở [CONTRIBUTING.md](CONTRIBUTING.md) và [onboarding](docs/onboarding.md).
- Tutorial dành cho baseline của tác giả nằm trong [reproduction](docs/reproduction/README.md).