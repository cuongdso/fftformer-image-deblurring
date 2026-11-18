# Ví dụ cấu hình

`evaluation-precomputed.json` là ví dụ C-06 cho một evaluation job trên prediction và ground truth đã có sẵn. File minh họa schema version, contract version, protocol, experiment ID, provenance và metadata identity tùy chọn. File hiện tham chiếu `../../tests/fixtures/ws2` cho cả prediction và target để kiểm tra validation đường dẫn; fixture này có ảnh grayscale/RGBA rất nhỏ nên không phải bộ ảnh RGB paired hợp lệ mặc định cho `run`. Người dùng cần thay hai đường dẫn bằng thư mục prediction và target RGB ghép cặp trước khi chạy evaluation thực.

Có thể kiểm tra schema mà không cần dependency computer-vision sau khi cài editable package:

```bash
python -m pip install --no-deps -e .
python -m fftformer_lab.evaluation.cli validate-config configs/examples/evaluation-precomputed.json
```

Ví dụ này không mô tả checkpoint, không khẳng định ảnh prediction do FFTformer tạo ra và không phải config benchmark GoPro hoặc RealBlur.
