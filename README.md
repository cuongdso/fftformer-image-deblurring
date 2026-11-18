# FFTformer Image Deblurring

Repository học thuật của nhóm cho bài tập lớn môn Xử lý ảnh, tập trung vào việc tìm hiểu và thực nghiệm phương pháp **FFTformer** cho bài toán khử mờ ảnh (*image deblurring*).

> Trạng thái hiện tại: WS1 và WS2 đã merge vào `main`; WS3-02 đã hoàn thành pipeline core; WS4-02/03/04 đã được Tech Lead review trên `feat/evaluation-integration` nhưng chưa merge. WS4 có metric/config/CLI đánh giá prediction có sẵn và integration WS2→WS3→WS4 bằng `MockModel`. Đây không phải tái hiện benchmark GoPro/RealBlur hay suy luận FFTformer đầy đủ. Xem [project status](docs/project-status.md).

## Bài toán Image Deblurring

Image deblurring hướng đến việc ước lượng ảnh rõ tiềm ẩn từ ảnh quan sát bị mờ. Một mô hình tạo ảnh mờ thường được viết là `B = I * K + N`, trong đó `B` là ảnh mờ, `I` là ảnh rõ tiềm ẩn, `K` là kernel làm mờ, `N` là nhiễu và `*` biểu thị phép chập. Đây là bài toán nghịch đảo khó do có thể mất thông tin và do kernel/nhiễu không luôn được biết trước.

## Tổng quan FFTformer

FFTformer (*Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring*) là công trình của Lingshun Kong và cộng sự tại CVPR 2023. Công trình khai thác định lý chập để xử lý một số phép toán Transformer trong miền tần số, với kiến trúc encoder–decoder bất đối xứng: Frequency-domain Self-Attention Solver (FSAS) được dùng ở decoder. Bài báo cũng đề xuất Discriminative Frequency-domain Feed-Forward Network (DFFN).

Các khái niệm đang được tìm hiểu:

- **FFT / IFFT:** biến đổi nhanh từ miền không gian sang miền tần số và phép biến đổi ngược để tái tạo tín hiệu.
- **Self-Attention:** cơ chế sử dụng các biểu diễn Query (Q), Key (K), Value (V) để mô hình hóa quan hệ giữa các đặc trưng.
- **FSAS:** ý tưởng ước lượng scaled dot-product attention bằng tích theo phần tử trong miền tần số, thay cho phép nhân ma trận trực tiếp trong miền không gian.
- **DFFN:** khối FFN dựa trên miền tần số với cơ chế gating, nhằm chọn lọc thông tin tần số thấp/cao phù hợp cho khôi phục ảnh rõ.

Các mô tả trên là ghi chú nhập môn dựa trên công trình gốc, không phải tuyên bố về mức độ triển khai hay tái hiện kết quả của nhóm.

## Mục tiêu nghiên cứu và thực nghiệm

- Hiểu bài toán khử mờ ảnh, biểu diễn miền không gian và miền tần số.
- Đọc, phân tích và ghi chú các thành phần của FFTformer.
- Chuẩn bị cấu trúc để tái hiện thực nghiệm một cách có kiểm chứng ở các milestone sau.
- Xây dựng ứng dụng minh họa chỉ khi phạm vi nghiên cứu đã sẵn sàng.

## Baseline và phạm vi mã nguồn

`third_party/FFTformer/` là Git submodule trỏ đến official implementation tại commit `01306b5d4072146a011c8c6e9d4927cf4607f1d5`. Baseline này chỉ dùng để đối chiếu và tái hiện; không sửa trực tiếp trong các task của nhóm. Mã nhóm sẽ phát triển độc lập trong `src/fftformer_lab/`. Chiến lược và ràng buộc giấy phép được ghi tại [ADR-0001](docs/decisions/0001-upstream-baseline.md).

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

## Bắt đầu

Đọc theo thứ tự [AGENTS.md](AGENTS.md), [INSTRUCTION.md](INSTRUCTION.md), [onboarding](docs/onboarding.md), [architecture](docs/architecture.md), [project status](docs/project-status.md), và tài liệu của workstream tương ứng trong `docs/modules/`. Trước khi thử baseline, đọc [upstream audit](docs/upstream-audit.md) và tutorial [reproduction](docs/reproduction/). Các hướng dẫn đó phân biệt nội dung đã audit với nội dung chưa chạy thực nghiệm.

Để dùng package nhóm từ một checkout mới mà không đặt `PYTHONPATH`, cài editable package:

```bash
python -m pip install --no-deps -e .
python -m fftformer_lab.evaluation.cli validate-config configs/examples/evaluation-precomputed.json
```

Lệnh `validate-config` chỉ cần package nhóm. Lệnh `run` để đo các cặp ảnh prediction/ground truth RGB cần PyTorch, NumPy, Pillow và OpenCV; nó không chạy suy luận FFTformer. Cài các dependency đó trong môi trường riêng theo nhu cầu, rồi dùng `python -m fftformer_lab.evaluation.cli run <config>`.
