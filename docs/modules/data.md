# WS2 — Dataset & Preprocessing

**Owner:** `phanthek2`; reviewer: TBD (WS1/WS3/WS4 + Tech Lead theo scope).
**Hiện trạng:** `M1-WS2-02` và `M1-WS2-03` đã có commit triển khai riêng trên
feature branch, chờ PR/review; chưa merge. Trạng thái tại [project status](../project-status.md).
Audit nguồn/layout WS2-01 và hai cặp ảnh ví dụ được ghi tại
[dataset audit](../research/ws2-dataset-audit.md).

Phạm vi và acceptance criteria ở [backlog](../backlog.md); interface chuẩn là
[C-02/C-03](../contracts.md). Không thay đổi các contract đó hoặc mã upstream.

## Đối chiếu baseline

Tại upstream pin `01306b5d4072146a011c8c6e9d4927cf4607f1d5`, `test.py` đọc
PIL input/target cùng filename, chuyển ảnh uint8 thành CHW `[0,1]` rồi dùng
DataLoader để thêm chiều batch. WS2 giữ quy tắc pixel của ảnh RGB uint8 này,
bổ sung validation và non-GT theo C-02/C-03 đã duyệt. Không import BasicSR.

WS3 tiếp tục sở hữu thêm chiều batch, device transfer, reflection padding bội 32
và crop về kích thước gốc. WS2 không pad, resize, crop, áp dụng EXIF orientation,
chỉnh gamma/saturation hoặc augmentation. Vì vậy các phép augmentation training
trong `basicsr/data/paired_image_dataset.py` không được áp dụng vào loader này.
Đây là loader theo contract M1, chưa phải pipeline huấn luyện/reproduction đầy đủ.

## Cài đặt

Từ repository root, dùng môi trường riêng. Với CPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[data]"
```

Extra `data` khai báo PyTorch, NumPy và Pillow. OpenCV là tùy chọn khi caller cần
đọc ảnh thành BGR array; `array_to_tensor(..., color_order="BGR")` là conversion
tường minh. `einops` còn cần khi chạy model WS1. Import package/manifest API
không bắt buộc cài các dependency ảnh; decode/normalization mới cần chúng.

## Public API

| API trong `fftformer_lab.data` | Vai trò |
| --- | --- |
| `read_image(path)` | Đọc một ảnh RGB 8-bit PNG/JPEG thành tensor CPU float32 CHW. |
| `array_to_tensor(array, *, color_order)` | Nhận HWC uint8 hoặc float32 đã ở `[0,1]`; caller bắt buộc chỉ rõ RGB/BGR; trả RGB CHW không chia sẻ bộ nhớ với array nguồn. |
| `validate_image_tensor(image)` | Kiểm kiểu tensor, dtype, shape, range và NaN/Inf; không sửa dữ liệu. |
| `SampleRecord` | `sample_id`, source path và optional target path, chưa decode ảnh. |
| `build_paired_manifest(input_root, target_root, recursive=False)` | Ghép đầy đủ 1:1 theo relative filename, bao gồm extension. |
| `build_single_manifest(input_root, recursive=False)` | Manifest cho ảnh không GT, không tự tìm/bịa target. |
| `save_manifest(records, json_path)` / `load_manifest(json_path)` | Ghi/đọc manifest schema v1 với relative paths, UTF-8 và LF khi ghi. |
| `ImageDataset(records)` | Sequence đọc lazy theo thứ tự `sample_id`; hỗ trợ index, slice và iteration. |
| `Sample` / `PreprocessingMetadata` | Đầu ra C-03 và policy metadata để consumer lưu thí nghiệm. |

File reader chỉ nhận PNG/JPG/JPEG (extension không phân biệt hoa/thường), RGB
3 kênh, 8-bit và một frame. Grayscale, palette, alpha, CMYK, PNG bit depth khác 8
và animated PNG bị từ chối rõ ràng. Muốn hỗ trợ các dạng này cần một conversion
policy được review; hiện không có conversion tự động. Float array không được tự
coi là dữ liệu `[0,255]`: float32 phải đã normalized và finite; uint16/float64 bị
từ chối. Tensor ảnh luôn giữ H/W gốc; các fixture 2x2 hợp lệ cho WS2 nhưng có thể
quá nhỏ cho reflection padding của WS3.

Manifest folder mặc định chỉ đọc trực tiếp trong root. `recursive=True` giữ tên
sequence trong ID, ví dụ `sequence/frame.png`, tránh gộp các frame trùng tên.
Các ID sort theo chuỗi nguyên bản, phân biệt hoa/thường, không sort theo thời gian
tạo file. Thiếu target, target dư, ID/source/target trùng hoặc extension không
hỗ trợ đều báo lỗi. Decode và kiểm kích thước cặp diễn ra khi đọc sample;
manifest thành công chưa chứng minh mọi file đều decode được. Đọc toàn bộ dataset
bằng iteration để kiểm tra mọi ảnh trước khi dùng.

## Ví dụ dùng paired và non-GT

```python
from pathlib import Path
from fftformer_lab.data import (
    ImageDataset, build_paired_manifest, build_single_manifest,
    load_manifest, save_manifest,
)

root = Path("data/ws2-dev")  # ví dụ dữ liệu local tự tạo, không phải GoPro
records = build_paired_manifest(root / "paired/input", root / "paired/target")
save_manifest(records, root / "paired-manifest.json")
paired = ImageDataset(load_manifest(root / "paired-manifest.json"))
sample = paired[0]
print(sample.sample_id, sample.image.shape, sample.target.shape)
print(sample.preprocessing_metadata.to_dict())

single = ImageDataset(build_single_manifest(root / "single/input"))
assert single[0].target is None
assert single[0].has_ground_truth is False
```

`data/ws2-dev` là thư mục ignored tùy chọn trên máy phát triển. Unit tests có
fixture riêng tại [tests/fixtures/ws2](../../tests/fixtures/ws2/README.md), không
phụ thuộc thư mục local này. Với official test layout, truyền hai root
`<dataset>/test/input` và `<dataset>/test/target` cho cùng builder. Với layout khác,
caller truyền roots tương ứng hoặc tạo `SampleRecord` có ID/paths tường minh.

Manifest JSON tối thiểu:

```json
{
  "schema_version": 1,
  "samples": [
    {
      "sample_id": "000.png",
      "source_path": "input/000.png",
      "target_path": "target/000.png",
      "has_ground_truth": true
    }
  ]
}
```

Non-GT dùng `target_path: null`, `has_ground_truth: false`. Loader kiểm schema,
duplicate JSON keys, duplicate IDs/paths và tồn tại file; resolve relative paths
theo vị trí JSON, độc lập CWD. Writer yêu cầu `.json`, không ghi đè ảnh nguồn;
cross-drive relative paths trên Windows báo lỗi. Format này chỉ là inventory của
WS2, không thay schema cấu hình/thí nghiệm thuộc WS4/C-06 và không tự chứng minh
nguồn/license/split của dataset ngoài.

`Sample` có `image`, `target`, `sample_id`, `source_path`, `target_path`,
`preprocessing_metadata`, `has_ground_truth`. `ImageDataset` truyền trực tiếp được
cho `InferencePipeline.run_batch(dataset)` của WS3; từng sample vẫn có metadata
riêng. Không dùng default PyTorch collation cho dataclass `Sample`; batch tensor,
training augmentation và orchestration mới phải được scope riêng.

## Lỗi và kiểm chứng

- `ImageReadError`: thiếu file, file hỏng, đường dẫn là directory hoặc decode lỗi.
- `ImageValidationError`: mode/channel/bit depth, dtype/layout/range/NaN/Inf sai.
- `DatasetValidationError`: root/manifest/schema/pair/duplicate/size/GT flag sai.
- `DataDependencyError`: thiếu dependency tại bước thật sự cần dùng.

Loader bổ sung sample ID và input/target paths vào lỗi, không thay bằng ảnh giả.
Source images chỉ được mở đọc; mỗi lần load tạo tensor mới, không augmentation
ngẫu nhiên và không cache tensor đã bị caller sửa.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/unit -p "test_data_*.py" -v
python -B -S -m unittest discover -s tests/unit -p "test_data_*.py" -v
```

Lệnh thứ hai kiểm manifest/import ở môi trường chỉ có standard library và skip
tests cần ảnh/tensor. Kết quả thực tế và giới hạn tại [handoff WS2](../research/ws2-handoff.md).
Đã kiểm loader trên hai cặp ảnh ví dụ Flower/Istanbul do tác giả GoPro công bố;
chưa kiểm toàn bộ GoPro/RealBlur split, LMDB, training hoặc benchmark chất lượng.
Có thể tái lập bằng `scripts/ws2_verify.py`; lệnh và phạm vi ở handoff.
