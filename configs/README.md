# Cấu hình nhóm

M0 chưa có cấu hình thực thi; trạng thái đó là lịch sử. Hiện repository có schema C-06 tại `evaluation.schema.json` và ví dụ chạy được về mặt cấu trúc tại `examples/evaluation-precomputed.json` cho CLI WS4.

Cấu hình evaluation xác định experiment, protocol, thư mục prediction/ground truth và đường dẫn metadata output. Mọi đường dẫn tương đối được resolve theo thư mục chứa config. Config không chứa secret, dataset lớn hoặc checkpoint; các identity tùy chọn chỉ được ghi khi caller có thông tin thật.

Từ checkout mới, cài package trước khi validate hoặc chạy CLI:

```bash
python -m pip install --no-deps -e .
python -m fftformer_lab.evaluation.cli validate-config configs/examples/evaluation-precomputed.json
```

Lệnh `run` chỉ đánh giá ảnh prediction RGB đã tồn tại với ground truth tương ứng. Nó cần runtime PyTorch, NumPy, Pillow và OpenCV, nhưng không chạy inference FFTformer hay tái hiện benchmark.
